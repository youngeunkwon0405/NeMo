# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import Literal, Optional

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from torch import Tensor
from torch.nn.parameter import Parameter

from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import (
    get_init_method,
    make_upper_case,
    reweighted_cross_entropy,
)


class HyenaModel(LanguageModule):
    """
    A class for the HyenaModel.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,  # Actually a hyena.HyenaConfig but avoid circular import
        hyena_stack_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        num_groups_hyena: int,
        num_groups_hyena_medium: int,
        num_groups_hyena_short: int,
        pre_process: bool = True,
        hybrid_override_pattern: str = None,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        post_layer_norm: bool = True,
        share_embeddings_and_output_weights: bool = True,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'rope',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        hyena_init_method: str = None,
        hyena_output_layer_init_method: str = None,
        remove_activation_post_first_layer: bool = True,
        add_attn_proj_bias: bool = True,
    ) -> None:
        super().__init__(config=transformer_config)

        self.transformer_config = transformer_config
        self.hyena_config = HyenaConfig()

        # Override HyenaConfig fields with user provided values
        self.hyena_config.num_groups_hyena = num_groups_hyena
        self.hyena_config.num_groups_hyena_medium = num_groups_hyena_medium
        self.hyena_config.num_groups_hyena_short = num_groups_hyena_short
        if hyena_init_method:
            self.transformer_config.init_method = get_init_method(
                hyena_init_method, self.transformer_config.num_layers, self.transformer_config.hidden_size
            )
        if hyena_output_layer_init_method:
            self.transformer_config.output_layer_init_method = get_init_method(
                hyena_output_layer_init_method, self.transformer_config.num_layers, self.transformer_config.hidden_size
            )

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.hyena_stack_spec: ModuleSpec = hyena_stack_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.hybrid_override_pattern = hybrid_override_pattern
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.post_layer_norm = post_layer_norm
        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.transformer_config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.transformer_config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.transformer_config.use_cpu_initialization,
            )

        self.decoder = build_module(
            hyena_stack_spec,
            self.transformer_config,
            self.hyena_config,
            hybrid_override_pattern=self.hybrid_override_pattern,
            max_sequence_length=self.max_sequence_length,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=self.post_layer_norm,
        )

        # In some Hyena species, the published checkpoint has identity activations after the first
        # MLP block, so we replicate this behavior in this implementation if remove_activation_post_first_layer.
        self.remove_activation_post_first_layer = remove_activation_post_first_layer
        if self.remove_activation_post_first_layer:
            if parallel_state.is_pipeline_first_stage(ignore_virtual=False):
                # Skip the first layer of the global model for this activation patch.
                start_idx = 1
            else:
                start_idx = 0
            mlp_no_act_config = deepcopy(self.decoder.layers[start_idx].mlp.config)
            mlp_no_act_config.activation_func = lambda x: x
            for hyena_layer in self.decoder.layers[start_idx:]:
                hyena_layer.mlp.activation_func = mlp_no_act_config.activation_func
                hyena_layer.mlp.config = mlp_no_act_config

        # In some Hyena species, the published checkpoint always has a bias in the linear projection
        # of the self-attention layers regardless of bias in other linear layers.
        self.add_attn_proj_bias = add_attn_proj_bias
        if self.add_attn_proj_bias and not self.config.add_bias_linear:
            for layer in self.decoder.layers:
                if isinstance(layer, TransformerLayer):
                    linear_proj = layer.self_attention.linear_proj
                    output_size = linear_proj.weight.shape[0]
                    linear_proj.bias = Parameter(
                        torch.empty(
                            output_size, dtype=linear_proj.config.params_dtype, device=linear_proj.weight.device
                        )
                    )
                    # Always initialize bias to zero.
                    with torch.no_grad():
                        linear_proj.bias.zero_()
                    setattr(linear_proj.bias, 'allreduce', True)
                    setattr(linear_proj, 'te_return_bias', True)
                    setattr(linear_proj, 'return_bias', True)
                    setattr(linear_proj, 'use_bias', True)
                    setattr(linear_proj.bias, 'sequence_parallel', linear_proj.config.sequence_parallel)

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                transformer_config.hidden_size,
                self.vocab_size,
                config=transformer_config,
                init_method=transformer_config.init_method,
                bias=self.config.add_bias_output,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
            )
            if self.config.add_bias_output:
                self.output_layer.bias.data.zero_()

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        loss_mask: Tensor = None,
        inference_context: Optional[BaseInferenceContext] = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tensor:
        """Forward pass for the HyenaModel."""
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, decoder_input, self.transformer_config, None
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # The following assert will currently fail when running inference.
        # Commented out for now.
        # TODO (duncan/rwaleffe): (1) confirm that the externally-generated
        #   attention mask is not needed and is ignored by the model in
        #   inference mode, (2) reduce the size of the externally-generated
        #   attention mask to prevent CPU OOM (as we did for training), (3)
        #   force the attention mask passed to the model in inference mode to
        #   be None, so this assert will succeed.
        # assert attention_mask is None, "The attention mask is ignored and should be set to None"

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        labels, lowercase_mask = make_upper_case(labels)
        loss = self.compute_language_model_loss(labels, logits)
        normalize_per_batch = True if self.config.to_upper == "normalized_weighted" else False
        loss = reweighted_cross_entropy(
            loss,
            (labels, loss_mask, lowercase_mask),
            lowercase_weight=self.hyena_config.lowercase_loss_reweighting,
            normalize_per_batch=normalize_per_batch,
        )
        return loss
