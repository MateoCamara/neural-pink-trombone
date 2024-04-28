import importlib
import os

import librosa
import numpy as np
import torch
import torchaudio
import yaml
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from scipy.io.wavfile import write
from tqdm import tqdm

from utils import utils


def load_model(model_name):
    models_module = importlib.import_module("models")
    model_class = getattr(models_module, model_name)
    return model_class(**config['model_params'], **config['exp_params'])

def load_dataloader(data_type):
    data_module = importlib.import_module("data")
    if data_type == "spectrogram":
        data_class = getattr(data_module, "SpectrogramDataloader")
    elif data_type == "embedding":
        data_class = getattr(data_module, "EmbeddingDataloader")
    else:
        raise ValueError(f"Data type {data_type} not recognized")

    return data_class

def denormalizar_mel_spec(mel_spec, max_spec_value=45, min_spec_value=-65):
    mel_spec = mel_spec * (max_spec_value - min_spec_value) + min_spec_value
    return mel_spec


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

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='./configs/config_betaVAESynth_1.yaml')

parser.add_argument('--output', '-o',
                    dest="output",
                    metavar='FILE',
                    help='path to the results directory',
                    default='../generated_samples/')

parser.add_argument('--number', '-n',
                    help='number of batch samples to generate',
                    type=int,
                    dest="number",
                    default=10)

parser.add_argument('--checkpoint', '-ckpt',
                    help='path to the checkpoint file',
                    type=str,
                    dest="checkpoint",
                    default="logs/betaVAESynth/version_1/checkpoints/epoch=99-step=250000.ckpt")


args = parser.parse_args()

config_path = Path(args.filename)
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = load_model(config['model_params']['name'])
state_dict = torch.load(args.checkpoint)['state_dict']
fixed_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(fixed_state_dict)
data_path = config['data_params']['data_path']
data_path = os.path.expanduser(data_path)

dataset = load_dataloader(config['data_params']['data_type'])

# train_dataset = dataset(os.path.join(data_path, "train"), os.path.join(data_path, "train.json"), **config['data_params'])
val_dataset = dataset(os.path.join(data_path, "test"), os.path.join(data_path, "test.json"), **config['data_params'])

# train_loader = DataLoader(train_dataset, batch_size=config['data_params']['batch_size'], shuffle=True, num_workers=config['data_params']['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=config['data_params']['num_workers'], shuffle=False)

for i, batch in tqdm(enumerate(val_loader)):
    batch[1] = batch[1][:, :, 0]
    regen_batch = model.forward(input=batch[0], params=batch[1])
    save_path = os.path.join(args.output, f"sample_{i}")

    if 'VAE' in config['model_params']['name']:
        write(save_path + "_pred.wav", 48000, convert_mel_to_audio(denormalizar_mel_spec(regen_batch[0][0][0])))
        write(save_path + "_true.wav", 48000, convert_mel_to_audio(denormalizar_mel_spec(regen_batch[1][0][0])))
        np.save(save_path + "_mu.npy", regen_batch[2][0].detach().cpu().numpy())
        np.save(save_path + "_logvar.npy", regen_batch[3][0].detach().cpu().numpy())
        np.save(save_path + "_paramspred.npy", denormalizar_params(regen_batch[4][0]).detach().cpu().numpy())
        np.save(save_path + "_paramstrue.npy", denormalizar_params(regen_batch[5][0]).detach().cpu().numpy())
    else:
        np.save(save_path + "_paramspred.npy", denormalizar_params(regen_batch[0][0]).detach().cpu().numpy())
        np.save(save_path + "_embedding.npy", regen_batch[1][0].detach().cpu().numpy())
        np.save(save_path + "_paramstrue.npy", denormalizar_params(regen_batch[2][0]).detach().cpu().numpy())

    # Save the regenerated samples

    if i == args.number:
        break