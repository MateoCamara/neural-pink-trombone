import copy

import torch
from torch import Tensor
from torchvision.utils import make_grid
from experiments.baseExperiment import BaseExperiment
from utils.interpolate_parameters import interpolate_params


class Experiment3(BaseExperiment):

    def __init__(self, vae_model, config) -> None:
        super().__init__(vae_model, config)

    def training_step(self, batch, batch_idx):
        embedding, parameters, previous_parameters = batch

        y_hat = self.forward(embedding, params=parameters)

        train_loss = self.model.loss_function(*y_hat,
                                              batch_idx=batch_idx,
                                              previous_parameters=previous_parameters)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.total_train_loss += train_loss['loss'].item()
        self.denom_train += 1

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        embedding, parameters, previous_parameters = batch

        y_hat = self.forward(embedding, params=parameters)

        val_loss = self.model.loss_function(*y_hat,
                                              batch_idx=batch_idx,
                                              previous_parameters=previous_parameters)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.total_val_loss += val_loss['loss'].item()
        self.denom_val += 1
