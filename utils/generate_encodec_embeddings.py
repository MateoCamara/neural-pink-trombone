import os

import torch
import numpy as np
import soundfile as sf
from encodec import EncodecModel
from tqdm import tqdm
import scipy.io.wavfile as wavfile

# Define the path for saving embeddings
save_path = '../../neural-pink-trombone-data/pt_encodec_dynamic_simplified_10changes'
pt_dataset_path = '../../neural-pink-trombone-data/pt_dataset_dynamic_simplified_10changes'

save_path = os.path.expanduser(save_path)
pt_dataset_path = os.path.expanduser(pt_dataset_path)


os.makedirs(save_path, exist_ok=True)
train_path = os.path.join(save_path, 'train')
test_path = os.path.join(save_path, 'test')
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)


def _load_codec_model(device):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(12.0)
    model.eval()
    model.to(device)
    return model



pt_dataset_train = os.path.join(pt_dataset_path, "train")
pt_dataset_test = os.path.join(pt_dataset_path, "test")

#create if doesn't exist
os.makedirs(pt_dataset_train, exist_ok=True)
os.makedirs(pt_dataset_test, exist_ok=True)


def process_dataset(dataset_path, folder):
    for audio_sample_name in tqdm(os.listdir(dataset_path)):
        if os.path.exists(os.path.join(folder, f'{audio_sample_name.split(".wav")[0]}.pt')):
            continue
        audio_sample_path = os.path.join(dataset_path, audio_sample_name)
        audio_sample, _ = sf.read(audio_sample_path)
        # Process each sample
        audio_tensor = torch.as_tensor(audio_sample).float().unsqueeze(0).unsqueeze(0)  # Add batch dimension
        embeddings = model_encodec.encode(audio_tensor)
        embeddings = embeddings[0][0]
        # Inverse process to get audio from embeddings
        arr = embeddings.to(device)
        emb = model_encodec.quantizer.decode(arr.transpose(0, 1))

        torch.save(emb, os.path.join(folder, f'{audio_sample_name.split(".wav")[0]}.pt'))


# Setup the model and device
device = torch.device('cpu')
model_encodec = _load_codec_model(device)

# Process and save embeddings
# process_dataset(pt_dataset_train, train_path)
process_dataset(pt_dataset_test, test_path)

print("Embedding generation and saving completed.")
