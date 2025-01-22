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

try:
    from nemo.export.vllm_hf_exporter import vLLMHFExporter
except Exception:
    raise Exception(
        "vLLM should be installed in the environment or import "
        "the vLLM environment in the NeMo FW container using "
        "source /opt/venv/bin/activate command"
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="Local path of the base model")
    parser.add_argument('--lora-model', required=True, type=str, help="Local path of the lora model")
    # parser.add_argument('--triton-model-name', required=True, type=str, help="Name for the service")
    args = parser.parse_args()

    lora_model_name = "lora_model"

    exporter = vLLMHFExporter()
    exporter.export(model=args.model, enable_lora=True)
    exporter.add_lora_models(lora_model_name=lora_model_name, lora_model=args.lora_model)

    print(
        "------------- Output: ", exporter.forward(input_texts=["How are you doing?"], lora_model_name=lora_model_name)
    )
