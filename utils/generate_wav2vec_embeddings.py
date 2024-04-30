import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torch

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
import scipy.io.wavfile as wavfile
import librosa

# Define the path for saving embeddings
save_path = '../../neural-pink-trombone-data/pt_wav2vec_simplified'
pt_dataset_path = '../../neural-pink-trombone-data/pt_dataset_simplified'

save_path = os.path.expanduser(save_path)
pt_dataset_path = os.path.expanduser(pt_dataset_path)

os.makedirs(save_path, exist_ok=True)
train_path = os.path.join(save_path, 'train')
test_path = os.path.join(save_path, 'test')
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)


pt_dataset_train = os.path.join(pt_dataset_path, "train")
pt_dataset_test = os.path.join(pt_dataset_path, "test")


def process_dataset(dataset_path, folder):
    for audio_sample_name in tqdm(os.listdir(dataset_path)):
        if os.path.exists(os.path.join(folder, f'{audio_sample_name.split(".wav")[0]}.pt')):
            continue
        with torch.no_grad():
            audio_sample_path = os.path.join(dataset_path, audio_sample_name)
            audio_sample, _ = sf.read(audio_sample_path)
            audio_sample_16k = librosa.resample(y=audio_sample, orig_sr=48_000,
                                                target_sr=16000)  # necesario para wav2vec, qué mal
            audio_tensor = processor(audio_sample_16k, return_tensors="pt", padding=False,
                                     sampling_rate=16000).input_values.to(device)
            # Process each sample
            output = model(audio_tensor)
            embeddings = output.extract_features.detach().cpu()
            # Inverse process to get audio from embeddings
            torch.save(embeddings, os.path.join(folder, f'{audio_sample_name.split(".wav")[0]}.pt'))


# Setup the model and device
device = torch.device('cuda:0')
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(device)

# Process and save embeddings
process_dataset(pt_dataset_train, train_path)
process_dataset(pt_dataset_test, test_path)

print("Embedding generation and saving completed.")
