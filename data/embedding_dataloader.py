import os
import json

import numpy as np
import torchaudio
import torch
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm


class EmbeddingDataloader(Dataset):
    bounds = [(75, 330), (0.5, 1), (12, 29), (2.05, 3.5), (0.6, 1.7), (20.0, 40.0), (0.5, 2), (0.5, 2.0)]
    min_spec_value = -65
    max_spec_value = 45

    def __init__(self, embbedings_dir, json_file):
        self.embbedings_dir = embbedings_dir

        with open(os.path.join(embbedings_dir, json_file), 'r') as f:
            self.metadata = json.load(f)

        # Crea una lista de los nombres de archivo (claves del JSON)
        self.audio_files = list(self.metadata.keys())

    def __len__(self):
        return len(self.metadata.keys())

    def __getitem__(self, idx):
        embbeding_name = self.audio_files[idx].split('.wav')[0] + '.pt'
        embbeding_path = os.path.join(self.embbedings_dir, embbeding_name)

        # Carga el archivo de audio
        # TODO: normalizar los embeddings?
        embbeding = torch.load(embbeding_path, map_location=torch.device('cpu'))

        # Obtiene los parámetros (etiquetas) asociados

        parameters = self.metadata[self.audio_files[idx]]
        parameters = self.normalizar_params(parameters)
        parameters = torch.tensor(parameters).float()

        return embbeding, parameters

    def normalizar_params(self, params):
        for i, (low, high) in enumerate(self.bounds):
            params[i] = (params[i] - low) / (high - low)
        return params

    def _filter_files_in_metadata(self, files_dir):
        files = [f.split(".")[0] for f in os.listdir(files_dir)]
        return {k: v for k, v in self.metadata.items() if k.split(".")[0] in files}
