import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import torch
from torch import optim
from aprox_1_vae.betaModel_simplified import BaseVAE
import lightning as L
from typing import List, TypeVar
from torchvision.utils import make_grid

torch.manual_seed(1)

Tensor = TypeVar('torch.tensor')

class VAEXperiment(L.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super().__init__()

        self.model = vae_model.double()
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.total_train_loss = 0
        self.total_val_loss = 0
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        spectrogram, params = batch
        self.curr_device = spectrogram.device

        results = self.forward(spectrogram, params=params)
        self.log_spectrogram_images(results[1], results[0], self.current_epoch)

        train_loss = self.model.loss_function(*results,
                                              M_N=self.params["kld_weight"],  # al_img.shape[0]/ self.num_train_imgs,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.total_train_loss += train_loss['loss'].item()

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        spectrogram, params = batch
        self.curr_device = spectrogram.device

        results = self.forward(spectrogram, params=params)
        val_loss = self.model.loss_function(*results,
                                              M_N=self.params["kld_weight"],  # al_img.shape[0]/ self.num_train_imgs,
                                              batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.total_val_loss += val_loss['loss'].item()

    def on_validation_epoch_end(self):
        # Print the total losses for this epoch
        print(f"Epoch {self.current_epoch}: Train Loss = {self.total_train_loss}, Validation Loss = {self.total_val_loss}")

        # Reset the total losses for the next epoch
        self.total_train_loss = 0
        self.total_val_loss = 0

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def log_spectrogram_images(self, input: Tensor, reconstructed: Tensor, step: int):
        images = torch.stack([input, reconstructed], dim=0)

        # Create a grid of images
        grid = make_grid(images[:, 0, 0, :, :].unsqueeze(1), nrow=2)

        # Add the grid of images to TensorBoard
        self.logger.experiment.add_image('Spectrogram and Reconstructed', grid, step)