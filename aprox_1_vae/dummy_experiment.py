import torch

from aprox_1_vae import BaseVAE
from aprox_1_vae.experiment import VAEXperiment


class DummyExperiment(VAEXperiment):
    def __init__(self, vae_model: BaseVAE, params: dict, spectrogram_shape: tuple) -> None:
        super().__init__(vae_model, params)
        self.dummy_spectrogram = torch.rand(spectrogram_shape).double().to(self.device)

    def training_step(self, batch, batch_idx):
        params = batch[1]
        self.curr_device = self.dummy_spectrogram.device

        results = self.forward(self.dummy_spectrogram, params=params)
        self.log_spectrogram_images(results[1], results[0], self.current_epoch)

        train_loss = self.model.loss_function(*results,
                                              M_N=self.params["kld_weight"],
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.total_train_loss += train_loss['loss'].item()

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        params = batch[1]
        self.curr_device = self.dummy_spectrogram.device

        results = self.forward(self.dummy_spectrogram, params=params)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params["kld_weight"],
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.total_val_loss += val_loss['loss'].item()