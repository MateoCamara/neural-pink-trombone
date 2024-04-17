import torch
from torch import optim
from aprox_1_vae.base import BaseVAE
import lightning as L
from typing import TypeVar
from torchvision.utils import make_grid

torch.manual_seed(1)

Tensor = TypeVar('torch.tensor')


class BaseExperiment(L.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 config: dict) -> None:
        super().__init__()

        self.model = vae_model.float()
        self.config = config
        self.total_train_loss = 0
        self.total_val_loss = 0
        self.denom_train = 0
        self.denom_val = 0
        self.sample = None

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        x, params = batch
        y_hat = self.forward(x, params=params)

        train_loss = self.model.loss_function(*y_hat,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.total_train_loss += train_loss['loss'].item()
        self.denom_train += 1

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        x, params = batch
        y_hat = self.forward(x, params=params)

        val_loss = self.model.loss_function(*y_hat,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.total_val_loss += val_loss['loss'].item()
        self.denom_val += 1

        if self.sample is None:
            self.sample = y_hat

    def on_validation_epoch_end(self):
        # Print the total losses for this epoch
        if (self.denom_train == 0) or (self.denom_val == 0):
            self.denom_train = 1
            self.denom_val = 1
        print(
            f"Epoch {self.current_epoch}: Train Loss = {self.total_train_loss / self.denom_train}, Validation Loss = {self.total_val_loss / self.denom_val}")

        # Reset the total losses for the next epoch
        self.total_train_loss = 0
        self.total_val_loss = 0
        self.denom_train = 0
        self.denom_val = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.config['LR'],
                               weight_decay=self.config['weight_decay'])
        return optimizer
