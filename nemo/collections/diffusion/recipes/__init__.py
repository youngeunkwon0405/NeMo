# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


from nemo.collections.diffusion.recipes import flux_12b, flux_535m
from nemo.collections.llm.recipes.log.default import default_log, default_resume
from nemo.collections.llm.recipes.optim import adam, sgd
from nemo.collections.llm.recipes.run.executor import torchrun

__all__ = [
    "adam",
    "sgd",
    "default_log",
    "default_resume",
    "flux_535m",
    "flux_12b",
    "torchrun",
]
