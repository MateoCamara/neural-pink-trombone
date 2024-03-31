import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import IterableDataset


class PTDataset(IterableDataset):
    bounds = [(75, 330), (0.5, 1), (12, 29), (2.05, 3.5), (0.6, 1.7), (20.0, 40.0), (0.5, 2), (0.5, 2.0)]
    f_bounds = [(75, 330), (0.5, 1), (8, 25), (1.3, 3.5), (0.5, 1.3), (20.0, 33.0), (0.5, 1.4), (0.5, 1.5)]

    def __init__(self, servidor_url, servidor_port, tamano_batch):
        self.servidor_url = servidor_url
        self.tamano_batch = tamano_batch
        self.servidor_port = servidor_port

    def __iter__(self):
        # Esta función debe conectarse al servidor y obtener los datos
        # Debe devolver un iterador de los datos que se utilizarán en cada batch
        # Por ejemplo, podría ser algo así:
        for _ in range(self.tamano_batch):
            # Simulamos una llamada al servidor y la obtención de un dato
            datos = self.obtener_datos_de_servidor()
            yield datos

    def obtener_datos_de_servidor(self):
        # Aquí debes implementar la lógica para llamar a tu servidor y obtener los datos
        # Por ejemplo:
        # response = requests.get(self.servidor_url)
        # data = response.json()
        # return data
        random_values = np.array([random.uniform(min_bound, max_bound) for min_bound, max_bound in self.bounds])
        json_params = self._convert_params_to_json(random_values, 1, lip_index)
        response = requests.post(f'{self.servidor_url}:{self.servidor_port}/pink-trombone', json=json_params)
        audio = self._process_received_audio(response)
        mel_spec = self._compute_mel_spectrogram(audio, 48000, 8000)[:, 20]
        mel_spec = mel_spec[np.newaxis, np.newaxis, :]
        return mel_spec, random_values

    def _convert_params_to_json(self, params, length, lip_index):
        param_names = [
            'frequency', 'voiceness', 'tongue_index', 'tongue_diam',
            'lip_diam', 'constriction_index', 'constriction_diam', 'throat_diam', 'lip_index'
        ]
        params_dict = {name: [value] for name, value in zip(param_names, params)}
        params_dict['lip_index'] = [lip_index]
        params_dict['length'] = length
        return params_dict

    def _process_received_audio(self, response, normalize=False):
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

    def _compute_mel_spectrogram(self, audio, sr, fmax):
        """
        Calcula el espectrograma MEL de un audio dado.
        """
        S = librosa.feature.melspectrogram(y=audio, sr=sr, fmax=fmax, n_mels=128)
        return librosa.power_to_db(S, ref=np.max)