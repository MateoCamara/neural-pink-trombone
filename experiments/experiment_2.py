import copy

import torch
from torch import Tensor
from torchvision.utils import make_grid
from experiments.baseExperiment import BaseExperiment
from utils.interpolate_parameters import interpolate_params


class Experiment2(BaseExperiment):

    def __init__(self, vae_model, config) -> None:
        super().__init__(vae_model, config)

    def training_step(self, batch, batch_idx):
        mel_spec, parameters, previous_parameters, waveform = batch

        y_hat = self.forward(mel_spec, params=parameters)

        train_loss = self.model.loss_function(*y_hat,
                                              M_N=self.config["kld_weight"],
                                              batch_idx=batch_idx,
                                              previous_parameters=previous_parameters)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.total_train_loss += train_loss['loss'].item()
        self.denom_train += 1

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        mel_spec, parameters, previous_parameters, waveform = batch

        y_hat = self.forward(mel_spec, params=parameters)

        val_loss = self.model.loss_function(*y_hat,
                                              M_N=self.config["kld_weight"],
                                              batch_idx=batch_idx,
                                              previous_parameters=previous_parameters)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.total_val_loss += val_loss['loss'].item()
        self.denom_val += 1

        if self.sample is None:
            self.sample = y_hat

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.sample is not None:
            self.log_spectrogram_images(self.sample[1], self.sample[0], self.current_epoch)
        self.sample = None

    def log_spectrogram_images(self, input: Tensor, reconstructed: Tensor, step: int):
        images = torch.stack([input[:, 1, :], reconstructed[:, 0, :]], dim=0)

        images = images.unsqueeze(0)
        images = images.repeat(100, 1, 1, 1, 1)

        # Create a grid of images
        grid = make_grid(images[:, 0, 0, :, :].unsqueeze(1), nrow=2)

        # Add the grid of images to TensorBoard
        self.logger.experiment.add_image('Spectrogram and Reconstructed', grid, step)
