from torch.utils.data import DataLoader, IterableDataset
import lightning as L
import numpy as np
import requests
import random
import librosa
import torch


class PTServidorDataset(IterableDataset):
    bounds = [(75, 330), (0.5, 1), (12, 29), (2.05, 3.5), (0.6, 1.7), (20.0, 40.0), (0.5, 2), (0.5, 2.0)]
    f_bounds = [(75, 330), (0.5, 1), (8, 25), (1.3, 3.5), (0.5, 1.3), (20.0, 33.0), (0.5, 1.4), (0.5, 1.5)]

    def __init__(self, servidor_url, servidor_port, tamano_batch, iteraciones=100, device=torch.device('cpu')):
        super().__init__()
        self.servidor_url = servidor_url
        self.tamano_batch = tamano_batch
        self.servidor_port = servidor_port
        self.iteraciones = iteraciones
        self.device = device

    def __iter__(self):
        mel_spec_batch = np.empty((0, 1, 128, 94))  # Adjust the shape according to your data
        random_values_batch = np.empty((0, 8))  # Adjust the shape according to your data
        contador = 0
        while contador < self.iteraciones:
            # Simula una llamada de servidor para obtener un dato.
            mel_spec, random_values = self.obtener_datos_de_servidor()
            mel_spec = self.normalizar_mel_spec(mel_spec)
            mel_spec_batch = np.concatenate((mel_spec_batch, mel_spec[np.newaxis, :]), axis=0)
            random_values_batch = np.concatenate((random_values_batch, random_values[np.newaxis, :]), axis=0)
            contador += 1

            if len(mel_spec_batch) == self.tamano_batch:
                yield (torch.from_numpy(mel_spec_batch).to(self.device), torch.from_numpy(random_values_batch).to(self.device))
                mel_spec_batch = np.empty((0, 1, 128, 94))  # Adjust the shape according to your data
                random_values_batch = np.empty((0, 8))

        # Make sure to return the last incomplete batch if it exists
        if len(mel_spec_batch) > 0:
            yield (
                torch.from_numpy(mel_spec_batch).to(self.device), torch.from_numpy(random_values_batch).to(self.device))

    def obtener_datos_de_servidor(self):
        # Aquí debes implementar la lógica para llamar a tu servidor y obtener los datos
        # Por ejemplo:
        # response = requests.get(self.servidor_url)
        # data = response.json()
        # return data
        random_values = np.array([random.uniform(min_bound, max_bound) for min_bound, max_bound in self.bounds])
        json_params = self._convert_params_to_json(random_values, 1, 43)
        response = requests.post(f'http://{self.servidor_url}:{self.servidor_port}/pink-trombone', json=json_params)
        audio = self._process_received_audio(response, normalize=True)
        mel_spec = self._compute_mel_spectrogram(audio, 48000, 8000)
        mel_spec = mel_spec[np.newaxis, :]
        return mel_spec, random_values

    @staticmethod
    def _convert_params_to_json(params, length, lip_index):
        param_names = [
            'frequency', 'voiceness', 'tongue_index', 'tongue_diam',
            'lip_diam', 'constriction_index', 'constriction_diam', 'throat_diam', 'lip_index'
        ]
        params_dict = {name: [value] for name, value in zip(param_names, params)}
        params_dict['lip_index'] = [lip_index]
        params_dict['length'] = length
        return params_dict

    @staticmethod
    def _process_received_audio(response, normalize=False):
        """
        Procesa el audio recibido del servidor.
        Necesitarás ajustar esta función según el formato en que recibas el audio.
        """
        waveform = np.array(list(response.json()['output'].values()))
        waveform = waveform[int(48_000 * 0.2):]  # Delete the warm up
        waveform[waveform == None] = 0
        waveform = waveform.astype(np.float64)
        if normalize:
            waveform = librosa.util.normalize(waveform)
        return waveform

    @staticmethod
    def _compute_mel_spectrogram(audio, sr, fmax):
        """
        Calcula el espectrograma MEL de un audio dado.
        """
        S = librosa.feature.melspectrogram(y=audio, sr=sr, fmax=fmax, n_mels=128)
        return librosa.power_to_db(S, ref=np.max)

    def normalizar_mel_spec(self, mel_spec):
        mel_spec = np.clip(mel_spec, -80, 0)
        mel_spec = (mel_spec + 80) / 80
        return mel_spec

    def save_audio_and_params(self, audio, params, index):
        librosa.output.write_wav(f"audio_{index}.wav", audio, 48000)
        with open(f"params_{index}.txt", "w") as f:
            f.write(str(params))