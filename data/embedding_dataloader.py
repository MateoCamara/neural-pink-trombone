import os
import json

import numpy as np
import torchaudio
import torch
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm


class EmbeddingDataloader(Dataset):
    bounds = [(75, 330), (0.5, 1), (12, 29), (2.05, 3.5), (0.6, 1.7), (20.0, 40.0), (0.5, 2), (0.5, 2.0)]
    min_spec_value = -65
    max_spec_value = 45

    def __init__(self, embbedings_dir, json_file, **kwargs):
        self.embbedings_dir = embbedings_dir

        self.audio_dir = kwargs.get('audio_path', None)

        with open(os.path.join(embbedings_dir, json_file), 'r') as f:
            self.metadata = json.load(f)

        # Crea una lista de los nombres de archivo (claves del JSON)
        self.audio_files = list(self.metadata.keys())

    def __len__(self):
        return len(self.metadata.keys())

    def __getitem__(self, idx):
        file_name = self.audio_files[idx]
        embbeding_name = file_name.split('.wav')[0] + '.pt'
        embbeding_path = os.path.join(self.embbedings_dir, embbeding_name)

        # Carga embeddings
        # TODO: normalizar los embeddings?
        embbeding = torch.load(embbeding_path)

        # Obtiene los parámetros (etiquetas) asociados

        parameters = self.metadata[self.audio_files[idx]]
        parameters = self.normalizar_params(parameters)
        parameters = torch.tensor(parameters).float()

        if self.audio_dir:
            if 'train' in embbeding_path:
                audio_path = os.path.join(self.audio_dir, "train", file_name)
            elif "test" in embbeding_path:
                audio_path = os.path.join(self.audio_dir, "test", file_name)
            else:
                raise ValueError("Audio dir not found")

            audio_path = os.path.expanduser(audio_path)

            waveform, sample_rate = torchaudio.load(audio_path)
            mel_spec = self._compute_mel_spectrogram(waveform, sample_rate, 8000, power=True)
            mel_spec = self.normalizar_mel_spec(mel_spec).float()
            return embbeding, parameters, mel_spec

        return embbeding, parameters

    def normalizar_params(self, params):
        for i, (low, high) in enumerate(self.bounds):
            params[i] = (params[i] - low) / (high - low)
        return params

    def _filter_files_in_metadata(self, files_dir):
        files = [f.split(".")[0] for f in os.listdir(files_dir)]
        return {k: v for k, v in self.metadata.items() if k.split(".")[0] in files}

    @staticmethod
    def _compute_mel_spectrogram(audio, sr, fmax, power=True):
        """
        Calcula el espectrograma MEL de un audio dado.
        """
        spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=128,
            f_max=fmax,
            n_fft=2048,
            window_fn=torch.hann_window,
            hop_length=512
        )
        S = spec_transform(audio)

        if power:
            db_transform = torchaudio.transforms.AmplitudeToDB('power', top_db=80.)
            S_dB = db_transform(S)
            return S_dB
        else:
            return S

    def normalizar_mel_spec(self, mel_spec):
        mel_spec = np.clip(mel_spec, self.min_spec_value, self.max_spec_value)
        mel_spec = (mel_spec - self.min_spec_value) / (self.max_spec_value - self.min_spec_value)
        return mel_spec