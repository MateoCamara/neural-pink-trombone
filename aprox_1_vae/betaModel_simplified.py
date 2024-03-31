import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar
from aprox_1_vae import BaseVAE

Tensor = TypeVar('torch.tensor')


class BetaVAE(BaseVAE):
    num_iter = 0

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, beta: int = 4, gamma: float = 1000.,
                 max_capacity: int = 25, Capacity_max_iter: int = 1e5, loss_type: str = 'B', **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]

        self.encoder = self.build_encoder(in_channels, hidden_dims)
        self.encoder_output_size = self.get_output_size(self.encoder, torch.randn(1, 1, 128, 94), False)
        self.encoder_output = self.build_encoder_output(self.encoder_output_size.numel())
        self.fc_mu = nn.Linear(self.encoder_output_size.numel(), self.latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_size.numel(), self.latent_dim)

        self.decoder_input = self.build_decoder_input(self.latent_dim, self.encoder_output_size.numel())
        self.decoder = self.build_decoder(hidden_dims[::-1], in_channels)
        self.final_layer = self.build_final_layer(hidden_dims[0], in_channels)

    def build_encoder(self, in_channels, hidden_dims):
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                                         nn.BatchNorm2d(h_dim), nn.LeakyReLU(0.1)))
            in_channels = h_dim
        return nn.Sequential(*modules)

    def build_encoder_output(self, output_size):
        return nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(output_size, output_size),
                             nn.BatchNorm1d(output_size), nn.LeakyReLU(),
                             )

    def build_decoder_input(self, latent_dim, output_size):
        return nn.Sequential(nn.Linear(latent_dim, output_size), nn.LeakyReLU())

    def build_decoder(self, hidden_dims, in_channels):
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                              nn.BatchNorm2d(h_dim), nn.LeakyReLU(0.1)))
            in_channels = h_dim
        return nn.Sequential(*modules)

    def build_final_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=1,
                               output_padding=1),
            nn.Sigmoid())

    def get_output_size(self, model, input_tensor):
        output = input_tensor
        for m in model.children():
            for n in m.children():
                output = n(output)
        return output

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = self.encoder_output(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder_output_size.shape[1], self.encoder_output_size.shape[2],
                             self.encoder_output_size.shape[3])
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']

        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
