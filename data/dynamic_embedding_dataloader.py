import copy
import os
import json

import numpy as np
import torchaudio
import torch
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from utils import utils


class DynamicEmbeddingDataloader(Dataset):
    bounds = utils.bounds
    min_spec_value = utils.min_spec_value
    max_spec_value = utils.max_spec_value

    def __init__(self, embbedings_dir, json_file, **kwargs):
        self.embbedings_dir = embbedings_dir

        self.audio_dir = kwargs.get('audio_path', None)

        with open(json_file, 'r') as f:
            self.metadata = json.load(f)

        self.metadata = {k: v[2:] for k, v in self.metadata.items() if v is not None}
        self.num_of_samples = len(self.metadata[list(self.metadata.keys())[0]][0])

        # Crea una lista de los nombres de archivo (claves del JSON)
        self.audio_files = list(self.metadata.keys())

    def __len__(self):
        return len(self.metadata.keys())

    def __getitem__(self, idx):
        file_name = self.audio_files[idx]
        embbeding_name = file_name.split('.wav')[0] + '.pt'
        embbeding_path = os.path.join(self.embbedings_dir, embbeding_name)

        random_index = np.random.randint(0, self.num_of_samples)

        # Carga embeddings
        # TODO: normalizar los embeddings?
        embbeding = torch.load(embbeding_path)
        if len(embbeding.size()) == 3:
            embbeding = embbeding[0]

        if 'encodec' in embbeding_path:
            embbeding = embbeding.T # qué asco! pero bueno, es lo que hay

        embbeding_interpolated = torch.tensor(self.interpolate_embeddings(embbeding, original_fps=embbeding.shape[0])).float()

        # Obtiene los parámetros (etiquetas) asociados

        parameters = copy.deepcopy([i[random_index] for i in self.metadata[file_name]])
        if random_index > 0:
            previous_parameters = copy.deepcopy([i[random_index-1] for i in self.metadata[file_name]])
            embbeding_interpolated = embbeding_interpolated[random_index-1:random_index+1, :]
        else:
            previous_parameters = copy.deepcopy(parameters)
            # concatenate the first mel spectrogram with itself
            embbeding_interpolated = torch.cat([embbeding_interpolated[0, :].unsqueeze(0)] * 2, dim=0)

        parameters = self.normalizar_params(parameters)
        parameters = torch.tensor(parameters).float()

        previous_parameters = self.normalizar_params(previous_parameters)
        previous_parameters = torch.tensor(previous_parameters).float()

        if parameters.min() < 0 or parameters.max() > 1:
            raise ValueError(f"Parameters have values below 0 or above 1: {parameters.min()}, {parameters.max()}")

        if self.audio_dir:
            audio_path = os.path.join(self.audio_dir, file_name)
            audio_path = os.path.expanduser(audio_path)

            waveform, sample_rate = torchaudio.load(audio_path)
            mel_spec = self._compute_mel_spectrogram(waveform, sample_rate, 8000, power=True)
            mel_spec = self.normalizar_mel_spec(mel_spec).float()

            if mel_spec.min() < 0 or mel_spec.max() > 1:
                raise ValueError(f"Mel spectrogram has values below 0 or above 1: {mel_spec.min()}, {mel_spec.max()}")

            return embbeding, parameters, mel_spec, previous_parameters, waveform

        return embbeding_interpolated, parameters, previous_parameters

    def normalizar_params(self, params):
        for i, (low, high) in enumerate(self.bounds):
            params[i] = (params[i] - low) / (high - low)
        return params

    def denormalizar_params(self, params):
        for i, (low, high) in enumerate(self.bounds):
            params[i] = params[i] * (high - low) + low
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

    def convert_mel_to_audio(self, mel_spec):
        # mel_spectrogram_torchaudio_amplitude = torch.pow(torch.tensor(10.0), mel_spec / 20.0).numpy()[0]

        audio_reconstructed = librosa.feature.inverse.mel_to_audio(
            mel_spec.numpy()[0],
            sr=48000,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window='hann',
            center=True,
            pad_mode='reflect',
            power=2.0,
            n_iter=32,  # Número de iteraciones para Griffin-Lim
            htk=True,
            fmin=0,
            fmax=8000
        )
        return audio_reconstructed

    def slerp(self, p0, p1, t):
        """Interpola esféricamente entre dos puntos p0 y p1 usando el factor t."""
        omega = np.arccos(np.clip(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)), -1, 1))
        sin_omega = np.sin(omega)
        if sin_omega == 0:
            # Co-linear points, use linear interpolation
            return (1 - t) * p0 + t * p1
        else:
            return np.sin((1.0 - t) * omega) / sin_omega * p0 + np.sin(t * omega) / sin_omega * p1

    def interpolate_embeddings(self, embeddings, original_fps, target_fps=94):
        """Interpola los embeddings al número de pasos por segundo objetivo."""
        if original_fps == target_fps:
            return embeddings

        times_original = np.linspace(0, 1, original_fps)
        times_target = np.linspace(0, 1, target_fps)

        # Inicializa la lista de embeddings interpolados
        interpolated_embeddings = np.zeros((target_fps, embeddings.shape[1]))

        # Calcula interpolaciones para cada paso de tiempo en el objetivo
        for i in range(target_fps):
            # Encuentra los índices de los puntos originalmente más cercanos
            idx = np.searchsorted(times_original, times_target[i]) - 1
            idx_next = min(idx + 1, original_fps - 1)

            # Calcula el factor de interpolación
            t = (times_target[i] - times_original[idx]) / (times_original[idx_next] - times_original[idx])

            # Interpola esféricamente entre los dos puntos más cercanos
            interpolated_embeddings[i] = self.slerp(embeddings[idx], embeddings[idx_next], t)

        return interpolated_embeddings

    # Ejemplo de uso:
    # Supongamos que `embeddings_wav2vec` es un array de numpy con los embeddings de wav2vec
    # embeddings_wav2vec = np.random.rand(49, 128)  # 49 pasos, embeddings de 128 dimensiones
    # interpolated = interpolate_embeddings(embeddings_wav2vec, original_fps=49)

