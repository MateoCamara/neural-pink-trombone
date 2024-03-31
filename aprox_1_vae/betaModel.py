import torch
from aprox_1_vae import BaseVAE
from torch.nn import functional as F
from typing import List, TypeVar
from torch import nn
from data.pt_data_loader import PTServidorDataset

Tensor = TypeVar('torch.tensor')


class BetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]
        hidden_dims_decoder = hidden_dims[::-1]
        in_channels_original = in_channels

        # Build Encoder

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels=hidden_dims[0],
                          kernel_size=5,
                          stride=2,
                          padding=1
                          ),
                nn.LeakyReLU(0.1))
        )

        in_channels = hidden_dims.pop(0)

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.1))
            )
            in_channels = h_dim

        h_dim = 256

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=1, stride=1, padding=1),
                nn.LeakyReLU(0.1))
        )

        self.encoder = nn.Sequential(*modules)
        self.encoder_output_size = self.print_sizes(self.encoder, torch.randn(1, 1, 128, 94), False)

        self.encoder_output = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.encoder_output_size.numel(), self.encoder_output_size.numel()),
            nn.BatchNorm1d(self.encoder_output_size.numel()),
            nn.LeakyReLU(),
        )

        self.fc_mu = nn.Linear(self.encoder_output_size.numel(), self.latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_size.numel(), self.latent_dim)

        # Decoders

        # Asumiendo que self.latent_dim es la dimensión del espacio latente
        # y que tenemos las mismas dimensiones iniciales de la imagen

        decoder_modules = []

        # Primero, una capa lineal para "expandir" el espacio latente
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, self.encoder_output_size.numel()),
            nn.LeakyReLU(),
        )

        h_dim = 256
        out_channels = in_channels

        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(h_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
            )
        )

        in_channels = hidden_dims_decoder.pop(0)

        print(hidden_dims_decoder)

        # Añadimos las capas ConvTranspose2d
        for i, h_dim in enumerate(hidden_dims_decoder):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels=h_dim,
                                       kernel_size=4, stride=2, padding=1,
                                       output_padding=1 if i != len(hidden_dims_decoder) - 1 else (1, 0)),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.1))
            )
            in_channels = h_dim

        # Finalmente, necesitamos ajustar la última capa para que coincida con los canales de salida deseados
        # Por ejemplo, si la imagen de salida es en escala de grises (1 canal) o color (3 canales)

        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=in_channels_original,
                               # Este debe ser el número de canales de la imagen original, por ejemplo, 3 para RGB
                               kernel_size=5,
                               stride=2,
                               padding=1,
                               output_padding=1
                               ),
            nn.Hardtanh()  # Usualmente se usa Sigmoid al final para normalizar los pixeles entre 0 y 1
        )

    @staticmethod
    def print_sizes(model, input_tensor, printear=True):
        output = input_tensor
        for m in model.children():
            for n in m.children():
                output = n(output)
                if printear:
                    print(n, output.shape)
        return output

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # self.print_sizes(self.encoder, input)

        result = self.encoder(input)

        result = self.encoder_output(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder_output_size.shape[1], self.encoder_output_size.shape[2],
                             self.encoder_output_size.shape[3])
        # self.print_sizes(self.decoder, result)
        result = self.decoder(result)

        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
