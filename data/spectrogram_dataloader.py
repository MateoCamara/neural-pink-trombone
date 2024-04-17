import os
import json

import numpy as np
import torchaudio
import torch
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm


class SpectrogramDataloader(Dataset):
    bounds = [(75, 330), (0.5, 1), (12, 29), (2.05, 3.5), (0.6, 1.7), (20.0, 40.0), (0.5, 2), (0.5, 2.0)]
    min_spec_value = -65
    max_spec_value = 45
    def __init__(self, audio_dir, json_file):
        self.audio_dir = audio_dir

        # Carga los metadatos desde el archivo JSON
        with open(json_file, 'r') as f:
            self.metadata = json.load(f)

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

        # Obtiene los parámetros (etiquetas) asociados
        parameters = self.metadata[audio_name]
        parameters = self.normalizar_params(parameters)
        parameters = torch.tensor(parameters).float()

        return mel_spec, parameters

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

    # def normalizar_mel_spec(self, mel_spec):
    #     mel_spec = np.clip(mel_spec, self.min_spec_value, self.max_spec_value)
    #     mel_spec = (mel_spec + ((self.min_spec_value + self.max_spec_value) / 2)) / ((self.max_spec_value - self.min_spec_value) / 2)
    #     return mel_spec

    def normalizar_mel_spec(self, mel_spec):
        mel_spec = np.clip(mel_spec, self.min_spec_value, self.max_spec_value)
        mel_spec = (mel_spec - self.min_spec_value) / (self.max_spec_value - self.min_spec_value)
        return mel_spec

    def normalizar_params(self, params):
        for i, (low, high) in enumerate(self.bounds):
            params[i] = (params[i] - low) / (high - low)
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
