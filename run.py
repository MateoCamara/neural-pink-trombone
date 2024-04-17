import importlib
import os

import yaml
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping


def load_model(model_name):
    models_module = importlib.import_module("models")
    model_class = getattr(models_module, model_name)
    return model_class(**config['model_params'])


def load_experiment(exp_name, model):
    exp_module = importlib.import_module("experiments")
    exp_class = getattr(exp_module, exp_name)
    return exp_class(model, config['exp_params'])


def load_dataloader(data_type):
    data_module = importlib.import_module("data")
    if data_type == "spectrogram":
        data_class = getattr(data_module, "SpectrogramDataloader")
    elif data_type == "embedding":
        data_class = getattr(data_module, "EmbeddingDataloader")
    else:
        raise ValueError(f"Data type {data_type} not recognized")

    return data_class


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='./configs/config_exp_encodec.yaml')

args = parser.parse_args()

config_path = Path(args.filename)
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['logging_params']['name'], )

model = load_model(config['model_params']['name'])
experiment = load_experiment(config['exp_params']['name'], model)
data_path = config['data_params']['data_path']
data_path = os.path.expanduser(data_path)

dataset = load_dataloader(config['data_params']['data_type'])

train_dataset = dataset(os.path.join(data_path, "train"), os.path.join(data_path, "train.json"))
val_dataset = dataset(os.path.join(data_path, "test"), os.path.join(data_path, "test.json"))

train_loader = DataLoader(train_dataset, batch_size=config['data_params']['batch_size'], shuffle=True, num_workers=config['data_params']['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=config['data_params']['batch_size'], num_workers=config['data_params']['num_workers'])

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
