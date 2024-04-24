from matplotlib import pyplot as plt
from torch.utils.data import IterableDataset
import os
import json
import numpy as np
import requests
import random
import librosa
import torch
import soundfile as sf
from tqdm import tqdm


class PTServidorDataset(IterableDataset):
    bounds = [(100, 100), (1, 1), (12, 29), (2.05, 3.5), (0.6, 1.7), (20.0, 40.0), (0.5, 2), (0.5, 2.0)]
    f_bounds = [(75, 330), (0.5, 1), (8, 25), (1.3, 3.5), (0.5, 1.3), (20.0, 33.0), (0.5, 1.4), (0.5, 1.5)]

    def __init__(self, servidor_url, servidor_port, tamano_batch, number_of_changes=1, iteraciones=100, device=torch.device('cpu')):
        super().__init__()
        self.servidor_url = servidor_url
        self.tamano_batch = tamano_batch
        self.servidor_port = servidor_port
        self.number_of_changes = number_of_changes
        self.iteraciones = iteraciones
        self.device = device

        self.tongue = Tongue()  # cutre pero ok

    def __iter__(self):
        mel_spec_batch = np.empty((0, 1, 128, 94))  # Adjust the shape according to your data
        random_values_batch = np.empty((0, 8))  # Adjust the shape according to your data
        contador = 0
        while contador < self.iteraciones:
            # Simula una llamada de servidor para obtener un dato.
            mel_spec, random_values = self.generate_random_audio()
            mel_spec = self.normalizar_mel_spec(mel_spec)
            mel_spec_batch = np.concatenate((mel_spec_batch, mel_spec[np.newaxis, :]), axis=0)
            random_values_batch = np.concatenate((random_values_batch, random_values[np.newaxis, :]), axis=0)
            contador += 1

            if len(mel_spec_batch) == self.tamano_batch:
                yield (
                    torch.from_numpy(mel_spec_batch).to(self.device),
                    torch.from_numpy(random_values_batch).to(self.device))
                mel_spec_batch = np.empty((0, 1, 128, 94))  # Adjust the shape according to your data
                random_values_batch = np.empty((0, 8))

        # Make sure to return the last incomplete batch if it exists
        if len(mel_spec_batch) > 0:
            yield (
                torch.from_numpy(mel_spec_batch).to(self.device), torch.from_numpy(random_values_batch).to(self.device))

    def generate_random_audio(self):
        waveform, random_values = self.obtener_datos_de_servidor()
        mel_spec = self._compute_mel_spectrogram(waveform, 48000, 8000)
        mel_spec = mel_spec[np.newaxis, :]
        return mel_spec, random_values

    def generate_specific_audio(self, params):
        json_params = self._convert_params_to_json(params, 1, 43)
        response = requests.post(f'http://{self.servidor_url}:{self.servidor_port}/pink-trombone', json=json_params)
        audio = self._process_received_audio(response, normalize=False)
        return audio

    def obtener_datos_de_servidor(self):
        random_values = np.array([np.random.uniform(min_bound, max_bound, size=self.number_of_changes).tolist() for min_bound, max_bound in self.bounds])

        if self.tongue:
            for i in range(self.number_of_changes):
                self.tongue.set_random_diameter()
                random_values[3, i] = self.tongue.diameter
                self.tongue.set_random_index_based_on_diam()
                random_values[2, i] = self.tongue.index
        json_params = self._convert_params_to_json(random_values, 1, 43)
        response = requests.post(f'http://{self.servidor_url}:{self.servidor_port}/pink-trombone', json=json_params)
        audio = self._process_received_audio(response, normalize=False)
        return audio, random_values

    def _convert_params_to_json(self, params, length, lip_index):
        param_names = [
            'frequency', 'voiceness', 'tongue_index', 'tongue_diam',
            'lip_diam', 'constriction_index', 'constriction_diam', 'throat_diam', 'lip_index'
        ]
        params_dict = {name: list(value) for name, value in zip(param_names, params)}
        params_dict['lip_index'] = [lip_index] * self.number_of_changes
        params_dict['length'] = length
        return params_dict

    @staticmethod
    def _process_received_audio(response, normalize=False):
        waveform = np.array(list(response.json()['output'].values()))
        waveform = waveform[int(48_000 * 0.2):]  # Delete the warm up
        waveform[waveform == None] = 0
        waveform = waveform.astype(np.float64)
        if normalize:
            waveform = librosa.util.normalize(waveform)
        return waveform

    @staticmethod
    def _compute_mel_spectrogram(audio, sr, fmax, power=True):
        """
        Calcula el espectrograma MEL de un audio dado.
        """
        S = librosa.feature.melspectrogram(y=audio, sr=sr, fmax=fmax, n_mels=128, htk=True, norm=None)
        if power:
            S_dB = librosa.power_to_db(S, ref=100)
            return S_dB
        else:
            return S

    def normalizar_mel_spec(self, mel_spec):
        mel_spec = np.clip(mel_spec, -80, 0)
        mel_spec = (mel_spec + 80) / 80
        return mel_spec

    def generate_records(self, output_dir, num_records=10):
        params_dict = {}
        num_digits = len(str(num_records))

        for i in tqdm(range(1, num_records + 1)):
            waveform, params = self.obtener_datos_de_servidor()
            audio_file_name = f"{str(i).zfill(num_digits)}.wav"
            self.save_audio(waveform, os.path.join(output_dir, audio_file_name))
            params_dict[audio_file_name] = params.tolist()

        with open(os.path.join(output_dir, 'params.json'), 'w') as f:
            json.dump(params_dict, f)

    def save_audio(self, waveform, audio_file_path):
        sf.write(audio_file_path, waveform, 48000)

    def find_min_max_values(self):
        min_max = {
            'min_mod': np.inf,
            'max_mod': -np.inf,
        }
        for i in range(100):
            waveform, _ = self.obtener_datos_de_servidor()
            mel_spec = self._compute_mel_spectrogram(waveform, 48000, 8000, power=True)
            min_mod = np.min(mel_spec)
            max_mod = np.max(mel_spec)
            if min_mod < min_max['min_mod']:
                min_max['min_mod'] = min_mod
            if max_mod > min_max['max_mod']:
                min_max['max_mod'] = max_mod
        print(min_max)
        return min_max


class Tongue:

    def __init__(self):
        self.min_diam = 2.05
        self.max_diam = 3.5

        self.diam_range = self.max_diam - self.min_diam
        self.diam_center = (self.max_diam + self.min_diam) / 2

        self.min_index = 12
        self.max_index = 29

        self.index_range = self.max_index - self.min_index
        self.index_center = (self.max_index + self.min_index) / 2

    def get_diam_interpolation(self, diameter_value):
        interpolation = (diameter_value - self.min_diam) / self.diam_range
        return np.clip(interpolation, 0, 1)

    def get_index_center_offset(self, interpolation):
        center_offset_diam = interpolation * self.index_range
        center_offset_radius = center_offset_diam / 2
        return center_offset_radius

    def set_diameter(self, diameter):
        self.diameter = diameter

    def set_index(self, index):
        self.index = index

    def set_random_diameter(self):
        self.diameter = random.uniform(self.min_diam, self.max_diam)


    def get_index_range_based_on_diam(self):
        diam_interpolation = self.get_diam_interpolation(self.diameter)
        inverted_diam_interpolation = 1 - diam_interpolation

        straightened_interpolation = inverted_diam_interpolation ** 0.58 - 0.2 * (inverted_diam_interpolation ** 2 - inverted_diam_interpolation)
        center_offset = self.get_index_center_offset(straightened_interpolation)

        return self.index_center - center_offset, self.index_center + center_offset

    def set_random_index_based_on_diam(self):
        self.index = random.uniform(*self.get_index_range_based_on_diam())

    def plot_tongue_positions(self):
        diameters = np.linspace(self.min_diam, self.max_diam, 100)
        index_ranges = []

        for d in diameters:
            self.set_diameter(d)
            index_min, index_max = self.get_index_range_based_on_diam()
            index_ranges.append((index_min, index_max))

        index_mins, index_maxs = zip(*index_ranges)  # Descomprime las tuplas en dos listas separadas

        plt.figure(figsize=(10, 6))
        plt.plot(diameters, index_mins, '-o', label='Minimum Index')
        plt.plot(diameters, index_maxs, '-o', label='Maximum Index')
        plt.fill_between(diameters, index_mins, index_maxs, color='gray', alpha=0.5, label='Index Range')
        plt.title('Tongue Position Variability')
        plt.xlabel('Diameter')
        plt.ylabel('Index')
        plt.legend()
        plt.grid(True)
        plt.show()
