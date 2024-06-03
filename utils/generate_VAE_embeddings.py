import copy
import importlib
import os

import numpy as np
import torch
import soundfile as sf
import torchaudio
import yaml
from tqdm import tqdm

# Define the path for saving embeddings
save_path = '../../neural-pink-trombone-data/pt_VAE_dynamic_simplified'
pt_dataset_path = '../../neural-pink-trombone-data/pt_dataset_dynamic_simplified'

save_path = os.path.expanduser(save_path)
pt_dataset_path = os.path.expanduser(pt_dataset_path)


os.makedirs(save_path, exist_ok=True)
train_path = os.path.join(save_path, 'train')
test_path = os.path.join(save_path, 'test')
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

pt_dataset_train = os.path.join(pt_dataset_path, "train")
pt_dataset_test = os.path.join(pt_dataset_path, "test")

#create if doesn't exist
os.makedirs(pt_dataset_train, exist_ok=True)
os.makedirs(pt_dataset_test, exist_ok=True)

def load_model(model_name, config):
    models_module = importlib.import_module("models")
    model_class = getattr(models_module, model_name)
    return model_class(**config['model_params'], **config['exp_params'])

def set_weights_to_model(model, state_dict_path, device='cpu'):
    state_dict = torch.load(state_dict_path, map_location=torch.device(device))['state_dict']
    fixed_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(fixed_state_dict)
    return model

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

def normalizar_mel_spec(mel_spec):
    mel_spec = np.clip(mel_spec, -65, 45)
    mel_spec = (mel_spec - -65) / (45 - -65)
    return mel_spec

def process_dataset(dataset_path, folder):
    for audio_sample_name in tqdm(os.listdir(dataset_path)):
        if os.path.exists(os.path.join(folder, f'{audio_sample_name.split(".wav")[0]}.pt')):
            continue
        audio_sample_path = os.path.join(dataset_path, audio_sample_name)
        audio_sample, _ = sf.read(audio_sample_path)
        mel_spec = _compute_mel_spectrogram(torch.Tensor(audio_sample), 48000, 8000, power=True)
        mel_spec = normalizar_mel_spec(mel_spec).float()

        mus = torch.zeros(mel_spec.shape[1], 64)
        for index in range(mel_spec.shape[1]):
            if index > 0:
                mel_spec_aux = mel_spec[:, index - 1:index + 1]
            else:
                mel_spec_aux = torch.cat([mel_spec[:, 0].unsqueeze(1)] * 2, dim=1)

            mel_spec_aux = mel_spec_aux.permute(1, 0)
            mel_spec_aux = mel_spec_aux.unsqueeze(0)  # Remove the first dimension

            mus[index] = model_VAE.forward(mel_spec_aux)[2][0]

        torch.save(mus, os.path.join(folder, f'{audio_sample_name.split(".wav")[0]}.pt'))


config_path = "config_betaVAE_dynamic_0.yaml"
with open(f'../configs/{config_path}', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model_name = 'betaVAE'

version_number = "version_1"
state_dict_path = os.path.join("../logs", model_name, version_number, "checkpoints", "last.ckpt")

model = load_model(config['model_params']['name'], config).to('cpu')
model_VAE = set_weights_to_model(model, state_dict_path, device='cpu')

# Process and save embeddings
process_dataset(pt_dataset_train, train_path)
process_dataset(pt_dataset_test, test_path)

print("Embedding generation and saving completed.")
