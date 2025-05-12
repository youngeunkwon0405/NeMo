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

import nemo_run as run
import pytest

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.nemotron import Nemotron3Config22B, NemotronModel
from nemo.collections.llm.recipes import nemotron3_22b_16k
from nemo.lightning import Trainer


class TestNemotron3_22B_16K:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return nemotron3_22b_16k

    def test_model(self, recipe_module):
        model = recipe_module.model()
        assert isinstance(model, run.Config)
        assert model.__fn_or_cls__ == NemotronModel

    def test_model_config_parameters(self, recipe_module):
        model = recipe_module.model()
        nemotron_config = model.config
        assert isinstance(nemotron_config, run.Config)
        assert nemotron_config.__fn_or_cls__ == Nemotron3Config22B
        assert nemotron_config.num_layers == 40
        assert nemotron_config.hidden_size == 6144
        assert nemotron_config.seq_length == 16384
        assert nemotron_config.num_attention_heads == 48

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == NemotronModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 16384
        assert recipe.data.global_batch_size == 32
        assert recipe.data.micro_batch_size == 1

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_valid_trainer_parallelism(self, recipe_module):
        trainer_config = recipe_module.pretrain_recipe().trainer

        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"

        assert trainer_config.strategy.expert_model_parallel_size == 1

        assert (
            trainer_config.strategy.tensor_model_parallel_size
            * trainer_config.strategy.pipeline_model_parallel_size
            * trainer_config.strategy.context_parallel_size
            * trainer_config.strategy.expert_model_parallel_size
            % trainer_config.devices
            == 0
        )
        assert (
            trainer_config.strategy.tensor_model_parallel_size
            * trainer_config.strategy.pipeline_model_parallel_size
            * trainer_config.strategy.context_parallel_size
            * trainer_config.strategy.expert_model_parallel_size
            / trainer_config.devices
            % trainer_config.num_nodes
            == 0
        )

        if trainer_config.strategy.pipeline_model_parallel_size != 1:
            assert trainer_config.strategy.pipeline_dtype is not None

        if trainer_config.strategy.tensor_model_parallel_size == 1:
            assert trainer_config.strategy.sequence_parallel is False
