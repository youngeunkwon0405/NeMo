# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Callable, Optional

import lightning.pytorch as pl
import nemo_run as run
import torch
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.llm.gpt.model.phi3mini import Phi3ConfigMini, Phi3Model
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.utils.exp_manager import TimingCallback

NAME = "phi3_mini_4k_instruct"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Phi3 Mini 4k instruct model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Phi3 mini 4k instruct model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=phi3_mini_4k_instruct ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(Phi3Model, config=run.Config(Phi3ConfigMini))


def trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for Phi3 mini 4k instruct model.

    This function sets up the distributed training strategy and other training parameters.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        callbacks (Optional[list[run.Config[Callback]]]): List of callback configurations.

    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.

    Examples:
        CLI usage:
            $ nemo llm pretrain trainer=phi3_mini_4k_instruct ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=2, num_gpus_per_node=8)
            >>> print(trainer_config)

    Note:
        For more information on distributed training strategies, refer to the
        NeMo documentation on multi-GPU and multi-node training.
    """
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        limit_test_batches=50,
        limit_val_batches=32,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=2000,
    )

    return trainer


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    tensor_parallelism: int = 1,
    num_gpus_per_node: int = 1,
    max_steps: int = 1168251,
    performance_mode: bool = False,
    fn: Callable = pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for phi3_mini_4k_instruct model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        performance_mode (bool): If true, enables optimizations for maximum performance.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory phi3_mini_4k_instruct
            $ nemo llm pretrain --factory "phi3_mini_4k_instruct(num_nodes=1, name='my_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="phi3_mini_4k_instruct", num_nodes=1)
            >>> print(recipe)
    """
    recipe = run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(MockDataModule, seq_length=4096, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )

    return recipe


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
    tensor_parallelism: int = 1,
    max_steps: int = 116825,
    peft_scheme: Optional[str] = 'lora',
    seq_length: Optional[int] = None,
    packed_sequence: Optional[bool] = None,
    performance_mode: bool = False,
) -> run.Partial:
    """
    Create a fine-tuning recipe for Phi3 mini-4k-instruct model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.
        seq_length (int): Maximum number of tokens per microbatch.
        packed_sequence (Optional[bool]): If true, fine-tuning sequences will be packed into batches up to the given
        maximum seq_length for better efficiency. By default, this value equals performance_mode.
        performance_mode (bool): If true, enables optimizations for maximum performance.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory phi3_mini_4k_instruct

        Python API usage:
            >>> recipe = finetune_recipe(name="phi3_mini_4k_instruct", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning.
    """
    # Default to unpacked data in normal mode and packed data in performance mode
    # once packing recipe is well tested, change this default to true
    if packed_sequence is None:
        packed_sequence = performance_mode

    # For unpacked sequence, most samples in SQuAD dataset are shorter than 2K
    if seq_length is None:
        seq_length = 4096 if packed_sequence else 2048

    recipe = default_finetune_recipe(
        model(), "microsoft/Phi-3-mini-4k-instruct", dir, name, num_nodes, num_gpus_per_node, packed_sequence
    )
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 1
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() in ['lora', 'dora']:
        recipe.peft = run.Config(PEFT_STR2CLS[peft_scheme.lower()])
        recipe.peft.dim = 8
        recipe.peft.alpha = 16
        recipe.optim.config.use_distributed_optimizer = False

        # some settings currently do not function correctly with LoRA
        recipe.model.config.cross_entropy_loss_fusion = False

        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    # Sequence length settings in the model and dataset must agree
    recipe.model.config.seq_length = seq_length
    recipe.data.seq_length = seq_length
    if packed_sequence:
        recipe.data.dataset_kwargs = {'pad_to_max_length': True}
        recipe.data.packed_sequence_specs = run.Config(PackedSequenceSpecs, packed_sequence_size=seq_length)

    return recipe
