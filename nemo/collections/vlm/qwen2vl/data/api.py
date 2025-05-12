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

import lightning.pytorch as pl

from nemo.collections.vlm.qwen2vl.data.mock import Qwen2VLMockDataModule
from nemo.collections.vlm.qwen2vl.data.preloaded import Qwen2VLPreloadedDataModule


def mock() -> pl.LightningDataModule:
    """Mock Qwen2-VL Data Module"""
    return Qwen2VLMockDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


def preloaded() -> pl.LightningDataModule:
    """Preloaded Qwen2-VL-like Data Module"""
    return Qwen2VLPreloadedDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


__all__ = ["mock", "preloaded"]
