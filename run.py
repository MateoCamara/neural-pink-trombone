import importlib
import os

import yaml
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import torch

torch.set_float32_matmul_precision('medium')

def load_model(model_name, config):
    models_module = importlib.import_module("models")
    model_class = getattr(models_module, model_name)
    return model_class(**config['model_params'], **config['exp_params'])


def load_experiment(exp_name, model, config):
    exp_module = importlib.import_module("experiments")
    exp_class = getattr(exp_module, exp_name)
    return exp_class(model, config['exp_params'])


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

def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='./configs/config_encodec_dynamic_0.yaml')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['logging_params']['name'], )

    model = load_model(config['model_params']['name'], config)
    experiment = load_experiment(config['exp_params']['name_exp'], model, config)
    data_path = config['data_params']['data_path']
    data_path = os.path.expanduser(data_path)

    dataset, json_file = load_dataloader(config['data_params']['data_type'])

    train_dataset = dataset(os.path.join(data_path, "train"), os.path.join(data_path, json_file[0]), **config['data_params'])
    val_dataset = dataset(os.path.join(data_path, "test"), os.path.join(data_path, json_file[1]), **config['data_params'])

    train_loader = DataLoader(train_dataset, batch_size=config['data_params']['batch_size'], shuffle=True, num_workers=config['data_params']['num_workers'], persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data_params']['batch_size'], num_workers=config['data_params']['num_workers'], persistent_workers=True)

    tb_logger.log_hyperparams(config)

    # data.setup()
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=config['exp_params']['patience'], verbose=True, mode='min')
    runner = L.Trainer(logger=tb_logger,
                       callbacks=[
                           LearningRateMonitor(),
                           ModelCheckpoint(save_top_k=2,
                                           dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                           monitor="val_loss",
                                           save_last=True),
                           early_stop_callback
                       ],
                       **config['trainer_params'])

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main() # aparentemente esto es necesario para windows