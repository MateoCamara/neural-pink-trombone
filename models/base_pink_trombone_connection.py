import numpy as np
import requests
import torch
import torchaudio

import utils.utils


class PinkTromboneConnection:
    param_bounds = np.array([(75, 330), (0.5, 1), (12, 29), (2.05, 3.5), (0.6, 1.7), (20.0, 40.0), (0.5, 2), (0.5, 2.0)])
    min_spec_value = -65
    max_spec_value = 45
    def __init__(self, servidor_url, servidor_port):
        self.servidor_url = servidor_url
        self.servidor_port = servidor_port

    def regenerate_audio_from_pred_params(self, params, audio_length=1.0):
        mel_specs = []
        for params_in_batch in params:

            denorm_params = self.denormalize_params(params_in_batch)
            json_params = self.convert_params_to_json(denorm_params, audio_length)
            response = self.send_params_to_pink_trombone(json_params)
            mel_spec = self.process_received_audio(response)
            mel_specs.append(mel_spec)

        return torch.stack(mel_specs).unsqueeze(1)

    def denormalize_params(self, params):
        denorm_params = []
        for i, (low, high) in enumerate(self.param_bounds):
            denorm_params.append(params[i] * (high - low) + low)
        return denorm_params

    def convert_params_to_json(self, params, audio_length):
        param_names = utils.utils.params_names
        params_dict = {name: [value] for name, value in zip(param_names, params)}
        params_dict['lip_index'] = [43]
        params_dict['length'] = audio_length
        return params_dict

    def send_params_to_pink_trombone(self, params):
        response = requests.post(f'http://{self.servidor_url}:{self.servidor_port}/pink-trombone', json=params)
        return response

    def process_received_audio(self, response):
        waveform = np.array(list(response.json()['output'].values()))
        waveform = waveform[int(48_000 * 0.2):]  # Delete the warm up
        waveform[waveform == None] = 0

        S = self._compute_mel_spectrogram(torch.Tensor(waveform), 48000, 8000)
        S_norm = self.normalizar_mel_spec(S)
        return S_norm

    def _compute_mel_spectrogram(self, audio, sr, fmax, power=True):
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