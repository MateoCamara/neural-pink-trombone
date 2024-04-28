import os
import json

import numpy as np
import torchaudio
import torch
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm
import copy

from utils import utils


class SpectrogramDataloader(Dataset):
    bounds = utils.bounds
    min_spec_value = utils.min_spec_value
    max_spec_value = utils.max_spec_value

    def __init__(self, audio_dir, json_file, **kwargs):
        self.audio_dir = audio_dir

        # Carga los metadatos desde el archivo JSON
        with open(json_file, 'r') as f:
            self.metadata = json.load(f)

        self.metadata = {k: v[2:] for k, v in self.metadata.items() if v is not None}

        # Crea una lista de los nombres de archivo (claves del JSON)
        self.audio_files = list(self.metadata.keys())

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_name = self.audio_files[idx]
        audio_path = os.path.join(self.audio_dir, audio_name)

        # Carga el archivo de audio
        waveform, sample_rate = torchaudio.load(audio_path)
        mel_spec = self._compute_mel_spectrogram(waveform, sample_rate, 8000, power=True)
        mel_spec = self.normalizar_mel_spec(mel_spec).float()

        # raise error if mel spec has values below 0 or above 1
        if mel_spec.min() < 0 or mel_spec.max() > 1:
            raise ValueError(f"Mel spectrogram has values below 0 or above 1: {mel_spec.min()}, {mel_spec.max()}")

        # Obtiene los parámetros (etiquetas) asociados
        parameters = copy.deepcopy(self.metadata[audio_name])
        parameters = self.normalizar_params(parameters)
        parameters = torch.tensor(parameters).float()

        if parameters.min() < 0 or parameters.max() > 1:
            raise ValueError(f"Parameters have values below 0 or above 1: {parameters.min()}, {parameters.max()}")

        return mel_spec, parameters, waveform

    @staticmethod
    def _compute_mel_spectrogram(audio, sr, fmax, power=True):
        """
        Calcula el espectrograma MEL de un audio dado.
        """
        spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            f_min=0,
            f_max=8000,
            n_mels=128,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode='reflect',
            mel_scale='htk'  # Ensure mel scale is 'htk' for compatibility
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

    def denormalizar_mel_spec(self, mel_spec):
        mel_spec = mel_spec * (self.max_spec_value - self.min_spec_value) + self.min_spec_value
        return mel_spec

    def normalizar_params(self, params):
        for i, (low, high) in enumerate(self.bounds):
            for j in range(len(params[i])):
                params[i][j] = (params[i][j] - low) / (high - low)
        return params

    def denormalizar_params(self, params):
        for i, (low, high) in enumerate(self.bounds):
            for j in range(len(params[i])):
                params[i][j] = params[i][j] * (high - low) + low
        return params

    def find_min_max_values(self):
        min_max = {
            'min_mod': np.inf,
            'max_mod': -np.inf,
        }

        for i in tqdm(range(100_000)):
            audio_name = self.audio_files[i]

            audio_path = os.path.join(self.audio_dir, audio_name)

            # Carga el archivo de audio
            waveform, sample_rate = torchaudio.load(audio_path)
            mel_spec = self._compute_mel_spectrogram(waveform, sample_rate, 8000, power=True)

            min_mod = mel_spec.min()
            max_mod = mel_spec.max()
            if min_mod < min_max['min_mod']:
                min_max['min_mod'] = min_mod
            if max_mod > min_max['max_mod']:
                min_max['max_mod'] = max_mod
        print(min_max)
        return min_max
