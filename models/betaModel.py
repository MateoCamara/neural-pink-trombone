import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar
from models.base import BaseVAE

Tensor = TypeVar('torch.tensor')


class BetaVAE(BaseVAE):
    num_iter = 0  # Variable estática global para llevar la cuenta de las iteraciones

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, beta: int = 4, **kwargs):
        super(BetaVAE, self).__init__()

        # Definición de variables internas
        self.latent_dim = latent_dim
        self.beta = beta

        # Inicialización de dimensiones ocultas si no se proporcionan
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]

        self.build_encoder(in_channels, hidden_dims)
        self.build_decoder(hidden_dims[::-1], in_channels)

        self.print_neural_graph()

    def build_encoder(self, in_channels, hidden_dims):
        """Construye la parte del codificador del VAE."""
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(h_dim) if h_dim != hidden_dims[0] else nn.Identity()
                ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.prepare_latent_variables()

    def calculate_output_size(self, model, input_tensor):
        """Calcula el tamaño de salida de un modelo dado un tensor de entrada."""
        with torch.no_grad():
            output = model(input_tensor)
        return output.size()

    def prepare_latent_variables(self):
        """Prepara las variables latentes y las capas para mu y log_var."""
        self.encoder_output_size = self.calculate_output_size(self.encoder, torch.randn(1, 1, 128, 94))
        flat_size = self.encoder_output_size.numel()
        self.encoder_output = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.3),
            nn.Linear(flat_size, flat_size),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(flat_size, self.latent_dim)
        self.fc_var = nn.Linear(flat_size, self.latent_dim)

    def build_decoder(self, hidden_dims, in_channels_original):
        """Construye la parte del decodificador del VAE."""
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, self.encoder_output_size.numel()),
            nn.ReLU(),
        )
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=h_dim, out_channels=hidden_dims[i + 1] if i + 1 < len(
                    hidden_dims) else in_channels_original,
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1 if i != len(hidden_dims) - 2 else (1, 0)),
                nn.ReLU() if h_dim != hidden_dims[-1] else nn.Sigmoid(),
                nn.BatchNorm2d(hidden_dims[i + 1]) if i + 1 < len(hidden_dims) else nn.Identity()
            ))
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor):
        """Codifica la entrada y devuelve los códigos latentes."""
        result = self.encoder(input)
        result = self.encoder_output(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor):
        """Decodifica los códigos latentes en la reconstrucción de la entrada."""
        z = self.decoder_input(z)
        z = z.view(-1, self.encoder_output_size[1], self.encoder_output_size[2], self.encoder_output_size[3])
        result = self.decoder(z)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        """Reparametrización para obtener z."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input: Tensor, **kwargs):
        """Propagación hacia adelante del modelo."""
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

        recons_loss = F.mse_loss(recons, input, reduction='sum')
        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        weighted_kld_loss = self.beta * kld_weight * kld_loss

        loss = recons_loss + weighted_kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': weighted_kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
