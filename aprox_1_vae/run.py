import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import yaml
import argparse
from pathlib import Path
from experiment import VAEXperiment
from dummy_experiment import DummyExperiment
from aprox_1_vae.betaModel import BetaVAE
import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from data.pt_data_loader import PTServidorDataset

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='./config.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )

model = BetaVAE(**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])
# experiment = DummyExperiment(model,
#                           config['exp_params'],
#                              (10, 1, 128, 94))

# data = [[1,1], [1,2]]
pt_train_dataloader = PTServidorDataset("127.0.0.1", "3000", config['data_params']['batch_size'], config['data_params']['iterations'])
pt_val_dataloader = PTServidorDataset("127.0.0.1", "3000", config['data_params']['batch_size'], int(config['data_params']['iterations']*.2))

# data.setup()
runner = L.Trainer(logger=tb_logger,
                   callbacks=[
                       LearningRateMonitor(),
                       ModelCheckpoint(save_top_k=2,
                                       dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                       monitor="val_loss",
                                       save_last=True),
                   ],
                   **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, train_dataloaders=pt_train_dataloader, val_dataloaders=pt_val_dataloader)
