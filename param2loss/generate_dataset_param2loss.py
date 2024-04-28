# Imports
import numpy as np
import torch
import os
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # Adds the parent directory of the calling script to the system path
from data import PTServidorDataset

# Parameters
params_json_file = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/params.json'
params_noisy_json_file = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/params_noisy.json'
wavs_dir = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/wavs'
wavs_noisy_dir = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/wavs_noisy'


# Import PTServidorDataset class
dataset = PTServidorDataset(servidor_url='127.0.0.1', servidor_port=3000, tamano_batch=1, iteraciones=1)

# Generate noisy parameters
dataset.generate_corrupted_params(params_json_file, params_noisy_json_file)
# Generate spectrograms
dataset.generate_records_param2loss(params_json_file, wavs_dir)
# Generate noisy spectrograms
dataset.generate_records_param2loss(params_noisy_json_file, wavs_noisy_dir)