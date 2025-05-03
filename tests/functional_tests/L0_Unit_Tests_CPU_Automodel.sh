# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
CUDA_VISIBLE_DEVICES="" NEMO_NUMBA_MINVER=0.53 coverage run -a --data-file=/workspace/.coverage --source=/workspace/ -m pytest tests/automodel tests/collections/llm/hf/ tests/collections/vlm/hf/ -m "not pleasefixme" --cpu --with_downloads --relax_numba_compat
