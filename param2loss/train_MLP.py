import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, TypeVar
import os
import numpy as np
import json

class CustomDataset(Dataset):

    def __init__(self, orig_audio_dir, noisy_audio_dir, orig_param_json, noisy_param_json):
        self.audio_files_original = [os.path.join(orig_audio_dir, f) for f in os.listdir(orig_audio_dir)]
        self.audio_files_noisy = [os.path.join(noisy_audio_dir, f) for f in os.listdir(noisy_audio_dir)]

        with open(orig_param_json, 'r') as f:
            self.param_dict = json.load(f)

        with open(noisy_param_json, 'r') as f:
            self.param_dict_noisy = json.load(f)

    def __len__(self):
        return min(len(self.audio_files_original), len(self.audio_files_noisy))

    def __getitem__(self, idx):
        # Waveforms
        waveform_orig, sample_rate1 = torchaudio.load(self.audio_files_original[idx])
        waveform_noisy, sample_rate2 = torchaudio.load(self.audio_files_noisy[idx])
        # Normalized mel spectrograns
        mel_spec_orig = self.normalizar_mel_spec(self._compute_mel_spectrogram(waveform_orig, sample_rate1, 8000, power=True))
        mel_spec_noisy = self.normalizar_mel_spec(self._compute_mel_spectrogram(waveform_noisy, sample_rate2, 8000, power=True))
        true_MSE = F.mse_loss(mel_spec_noisy, mel_spec_orig, reduction='sum')
        # PT parameters
        params_orig = np.concatenate(self.param_dict[os.path.split(self.audio_files_original[idx])[-1]])
        params_noisy = np.concatenate(self.param_dict_noisy[os.path.split(self.audio_files_original[idx])[-1]])
        params_vector = torch.from_numpy(np.concatenate((params_orig,params_noisy)))

        return true_MSE, params_vector
    
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
        mel_spec = np.clip(mel_spec, -80, 0)
        mel_spec = (mel_spec + 80) / 80
        return mel_spec

class CustomLoss(torch.nn.Module):

    def forward(self, true_mse, estimated_mse):
        loss = F.mse_loss(true_mse, estimated_mse)
        return loss

class Param2loss_MLP(pl.LightningModule):

    num_iter = 0  # Variable estática global para llevar la cuenta de las iteraciones

    def __init__(self, in_channels: int = 16, out_channels: int = 1, hidden_dims: List = None, **kwargs):
        super(Param2loss_MLP, self).__init__()
        # Definición de variables internas
        self.out_channels = out_channels
        # Inicialización de dimensiones ocultas si no se proporcionan
        if hidden_dims is None:
            hidden_dims = [32, 16, 8, 4]
        # Construye red MLP
        self.build_network(in_channels, out_channels, hidden_dims)

    def build_network(self, in_channels, out_channels, hidden_dims):
        """Construye la red MLP"""
        # Input and hidden layers
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=h_dim)
                ))
            in_channels = h_dim
        # Output layer
        modules.append(nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=out_channels),
                nn.ReLU() # Final ReLU activation: output is nonnegatve (loss proxy)
                ))

        self.network = nn.Sequential(*modules)

    def forward(self, input, **kwargs):
        """Propagación hacia adelante del modelo."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        input = input.to(torch.float32)

        return self.network(input.to(device))

    def training_step(self, batch, batch_idx):
        true_MSE, params_vector = batch
        estimated_MSE = self.forward(params_vector)
        loss = CustomLoss()(true_MSE, torch.squeeze(estimated_MSE))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


## MAIN ##

# Paths
params_json_file = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/params.json'
params_noisy_json_file = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/params_noisy.json'
wavs_dir = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/wavs'
wavs_noisy_dir = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/wavs_noisy'

# TODO
# Split data in train, validation and test (different directories)


# Load data
train_dataset = CustomDataset(wavs_dir, wavs_noisy_dir, params_json_file, params_noisy_json_file)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=5)

# Load model
model = Param2loss_MLP()
trainer = pl.Trainer(max_epochs=10)

# Launch training
trainer.fit(model, train_dataloader)

# Save model weights
torch.save(model.state_dict(), f"./param2loss/nn_weights/model.pth")