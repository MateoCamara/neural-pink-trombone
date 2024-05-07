import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from scipy.io.wavfile import write
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from visualizations.visualization_utils import load_model, load_dataloader, denormalizar_mel_spec, convert_mel_to_audio, \
    denormalizar_params

NUMERO_DE_EJEMPLOS = 50

all_available_configs = os.listdir('../configs')

for config_path in all_available_configs:
    print(config_path)
    if config_path == 'config_betaVAE_0.yaml':
        continue

    with open(f'../configs/{config_path}', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model_name = '_'.join(config_path.split('_')[1:-1])
    if 'PT' in model_name:
        continue
    version_number = "version_" + config_path.split('_')[-1].split('.')[0]

    state_dict_path = os.path.join("../logs", model_name, version_number, "checkpoints", "last.ckpt")
    if not os.path.exists(state_dict_path):
        continue

    model = load_model(config['model_params']['name'], config).to('cpu')
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))['state_dict']
    fixed_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(fixed_state_dict)

    data_type = config['data_params']['data_type']
    data_path = os.path.join('..', config['data_params']['data_path'])
    data_path = os.path.expanduser(data_path)
    dataset, json_file = load_dataloader(data_type)
    val_dataset = dataset(os.path.join(data_path, "test"), os.path.join(data_path, json_file[1]), **config['data_params'])
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=False)

    for i, batch in tqdm(enumerate(val_loader)):
        regen_batch = model.forward(input=batch[0], params=batch[1])
        save_dir = os.path.join("../../generated_samples/", model_name + "_" + version_number)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"sample_{i}")
        os.makedirs(save_path, exist_ok=True)

        if data_type == 'spectrogram_dynamic':
            np.save(os.path.join(save_path, "specpred.npy"), denormalizar_mel_spec(regen_batch[0][0][0]).detach().cpu().numpy())
            np.save(os.path.join(save_path, "spectrue.npy"), denormalizar_mel_spec(regen_batch[1][0][1]).detach().cpu().numpy())
            np.save(os.path.join(save_path, "mu.npy"), regen_batch[2][0].detach().cpu().numpy())
            np.save(os.path.join(save_path, "logvar.npy"), regen_batch[3][0].detach().cpu().numpy())
            np.save(os.path.join(save_path, "paramspred.npy"), denormalizar_params(regen_batch[4][0]).detach().cpu().numpy())
            np.save(os.path.join(save_path, "paramstrue.npy"), denormalizar_params(regen_batch[5][0]).detach().cpu().numpy())
        elif data_type == 'embedding_dynamic':
            np.save(os.path.join(save_path, "paramspred.npy"), denormalizar_params(regen_batch[0][0]).detach().cpu().numpy())
            np.save(os.path.join(save_path, "emb.npy"), regen_batch[1][0][1].detach().cpu().numpy())
            np.save(os.path.join(save_path, "paramstrue.npy"), denormalizar_params(regen_batch[2][0]).detach().cpu().numpy())
        elif data_type == 'embedding':
            np.save(os.path.join(save_path, "paramspred.npy"),
                    denormalizar_params(regen_batch[0][0]).detach().cpu().numpy())
            np.save(os.path.join(save_path, "emb.npy"), regen_batch[1][0, :, :].detach().cpu().numpy())
            np.save(os.path.join(save_path, "paramstrue.npy"),
                    denormalizar_params(regen_batch[2][0, :, 0]).detach().cpu().numpy())
        elif data_type == 'spectrogram':
            np.save(os.path.join(save_path, "specpred.npy"),
                    denormalizar_mel_spec(regen_batch[0][0]).detach().cpu().numpy())
            np.save(os.path.join(save_path, "spectrue.npy"),
                    denormalizar_mel_spec(regen_batch[1][0]).detach().cpu().numpy())
            np.save(os.path.join(save_path, "mu.npy"), regen_batch[2][0].detach().cpu().numpy())
            np.save(os.path.join(save_path, "logvar.npy"), regen_batch[3][0].detach().cpu().numpy())
            np.save(os.path.join(save_path, "paramspred.npy"),
                    denormalizar_params(regen_batch[4][0]).detach().cpu().numpy())
            np.save(os.path.join(save_path, "paramstrue.npy"),
                    denormalizar_params(regen_batch[5][0, :, 0]).detach().cpu().numpy())

        if i == NUMERO_DE_EJEMPLOS - 1:
            break
