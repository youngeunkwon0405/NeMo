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

from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_matrices

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    get_hidden_length_from_sample_length,
    speaker_to_target,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseAudioToSpeechE2ESpkDiarDataset(torch.utils.data.Dataset):
    """
    This dataset is a Lhotse version of diarization dataset in audio_to_diar_label.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define the output types of the dataset."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'targets': NeuralType(('B', 'T', 'N'), LabelsType()),
            'target_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, cfg):
        super().__init__()
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = int(
            self.cfg.get('window_stride', 0.01) * self.cfg.get('sample_rate', 16000)
        )  # 160 samples for every 1ms by default
        self.num_mel_frame_per_target_frame = int(self.cfg.get('subsampling_factor', 8))
        self.spk_tar_all_zero = self.cfg.get('spk_tar_all_zero', False)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        speaker_activities = []
        for cut in cuts:
            speaker_activity = speaker_to_target(
                a_cut=cut,
                num_speakers=self.num_speakers,
                num_sample_per_mel_frame=self.num_sample_per_mel_frame,
                num_mel_frame_per_asr_frame=self.num_mel_frame_per_target_frame,
                spk_tar_all_zero=self.spk_tar_all_zero,
                boundary_segments=True,
            )
            speaker_activities.append(speaker_activity)
        targets = collate_matrices(speaker_activities).to(audio.dtype)
        target_lens_list = []
        for audio_len in audio_lens:
            target_fr_len = get_hidden_length_from_sample_length(
                audio_len, self.num_sample_per_mel_frame, self.num_mel_frame_per_target_frame
            )
            target_lens_list.append(target_fr_len)
        target_lens = torch.tensor(target_lens_list)

        return audio, audio_lens, targets, target_lens
