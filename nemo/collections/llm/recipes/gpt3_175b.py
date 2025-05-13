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
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model import GPTConfig175B, GPTModel
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048,
)
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "gpt3_175b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a GPT3 175B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the GPT3 175B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=gpt3_175b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(GPTModel, config=run.Config(GPTConfig175B))


def trainer(
    tensor_parallelism: int = 4,
    pipeline_parallelism: int = 8,
    pipeline_parallelism_type: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = 6,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    num_nodes: int = 64,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for GPT3 175B model.

    This function sets up the distributed training strategy optimized for the large 175B model.

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
            $ nemo llm pretrain trainer=gpt3_175b ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=64, num_gpus_per_node=8)
            >>> print(trainer_config)

    Note:
        This configuration uses extensive parallelism to handle the large model size efficiently.
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
    num_gpus_per_node: int = 8,
    performance_mode: bool = False,
    fn: Callable = pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for GPT3 175B model.

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
            $ nemo llm pretrain --factory gpt3_175b
            $ nemo llm pretrain --factory "gpt3_175b(num_nodes=64, name='my_175b_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="gpt3_175b_pretrain", num_nodes=64)
            >>> print(recipe)

    Note:
        This recipe is optimized for the large 175B model and requires significant computational resources.
    """
    recipe = run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(MockDataModule, seq_length=2048, global_batch_size=2048, micro_batch_size=2),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=0.9e-4),
        resume=default_resume(),
    )

    if performance_mode:
        recipe = pretrain_performance_optimizations(recipe)

    return recipe


def pretrain_performance_optimizations(recipe: run.Partial) -> run.Partial:
    """
    Create a performance-optimized pre-training recipe for GPT3 175B model.

    This method enables performance optimizations that may not be suitable for all use cases.
    It builds upon the standard pre-training recipe and adds additional performance enhancements.

    Args:
        recipe (run.Partial): Base pre-train recipe to which performance optimizations will be added

    Returns:
        run.Partial: Partial configuration for performance-optimized pre-training.

    Note:
        Use this method with caution and only when you need maximum performance.
        It may not be suitable for all hardware configurations or use cases.
    """

    if not recipe.trainer.callbacks:
        recipe.trainer.callbacks = []

    garbage_collection_callback = run.Config(
        GarbageCollectionCallback,
        gc_interval_train=100,
        gc_interval_val=100,
    )
    mcomm_overlap_callback = run.Config(
        MegatronCommOverlapCallback,
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048,
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=50,
        # 'overlap_param_gather_with_optimizer_step' is set automatically. Added here for user's knowledge
        overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
    )
    recipe.trainer.callbacks.extend(
        [
            garbage_collection_callback,
            mcomm_overlap_callback,
        ]
    )

    recipe.trainer.plugins.grad_reduce_in_fp32 = False
    recipe.optim.config.use_precision_aware_optimizer = False

    # TODO: Remove after functional error with this enabled is fixed
    recipe.model.config.use_transformer_engine_full_layer_spec = False

    return recipe
