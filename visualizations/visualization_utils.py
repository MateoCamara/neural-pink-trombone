import importlib

import librosa
import torch

from utils import utils


def load_model(model_name, config):
    models_module = importlib.import_module("models")
    model_class = getattr(models_module, model_name)
    return model_class(**config['model_params'], **config['exp_params'])

def set_weights_to_model(model, state_dict_path, device='cpu'):
    state_dict = torch.load(state_dict_path, map_location=torch.device(device))['state_dict']
    fixed_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(fixed_state_dict)
    return model

def load_dataloader(data_type):
    data_module = importlib.import_module("data")
    json_file = ['train.json', 'test.json']
    if data_type == "spectrogram":
        data_class = getattr(data_module, "SpectrogramDataloader")
    elif data_type == "embedding":
        data_class = getattr(data_module, "EmbeddingDataloader")
    elif data_type == "spectrogram_dynamic":
        data_class = getattr(data_module, "DynamicSpectrogramDataloader")
        json_file = ['train_interpolated.json', 'test_interpolated.json']
    elif data_type == "embedding_dynamic":
        data_class = getattr(data_module, "DynamicEmbeddingDataloader")
        json_file = ['train_interpolated.json', 'test_interpolated.json']
    else:
        raise ValueError(f"Data type {data_type} not recognized")

    return data_class, json_file

def denormalizar_mel_spec(mel_spec, max_spec_value=45, min_spec_value=-65):
    mel_spec = mel_spec * (max_spec_value - min_spec_value) + min_spec_value
    return mel_spec


def normalizar_params(params):
    for i, (low, high) in enumerate(utils.bounds):
        params[i] = (params[i] - low) / (high - low)
    return params

def denormalizar_params(params):
    for i, (low, high) in enumerate(utils.bounds):
        params[i] = params[i] * (high - low) + low
    return params


def convert_mel_to_audio(mel_spec):
    # mel_spectrogram_torchaudio_amplitude = torch.pow(torch.tensor(10.0), mel_spec / 20.0).numpy()[0]

    audio_reconstructed = librosa.feature.inverse.mel_to_audio(
        mel_spec.detach().numpy(),
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