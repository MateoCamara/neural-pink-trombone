import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar

import utils.utils
from models.base_pink_trombone_connection import PinkTromboneConnection
from models.base import BaseVAE

Tensor = TypeVar('torch.tensor')


class BetaVAESynth(BaseVAE):
    num_iter = 0  # Variable estática global para llevar la cuenta de las iteraciones

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, beta: int = 4, beta_params: list = [],
                 num_synth_params: int = 8, hidden_dims_synth_stage: List = None, params_weight: int = 1, **kwargs):
        super().__init__()

        # Definición de variables internas
        self.latent_dim = latent_dim
        self.beta = beta
        self.num_synth_params = num_synth_params
        self.beta_params = beta_params
        self.params_weight = params_weight

        # Inicialización de dimensiones ocultas si no se proporcionan
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]

        if hidden_dims_synth_stage is None:
            hidden_dims_synth_stage = [int(self.latent_dim / 2), int(self.latent_dim / 4), int(self.latent_dim / 8)]

        if beta_params is None:
            self.beta_params = [1] * num_synth_params

        self.build_encoder(in_channels, hidden_dims)
        self.build_synth_stage(hidden_dims_synth_stage)
        self.build_decoder(hidden_dims[::-1], in_channels)

        self.print_neural_graph()

        self.use_pink_trombone = kwargs.get('use_pink_trombone', False)

        if self.use_pink_trombone:
            self.regen_weight = kwargs.get('regen_weight', 1.0)
            self.pink_trombone_connection = PinkTromboneConnection(kwargs.get('pt_server', '127.0.0.1'), kwargs.get('pt_port', 3000))

    def build_encoder(self, in_channels, hidden_dims):
        """Construye la parte del codificador del VAE."""
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(h_dim) if h_dim != hidden_dims[0] else nn.ReLU()
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.prepare_latent_variables()

    def calculate_output_size(self, model, input_tensor):
        """Calcula el tamaño de salida de un modelo dado un tensor de entrada."""
        with torch.no_grad():
            for module in model:
                input_tensor = module(input_tensor)
        return input_tensor.size()

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
                # nn.BatchNorm2d(hidden_dims[i + 1]) if i + 1 < len(hidden_dims) else nn.Identity()
            ))
        self.decoder = nn.Sequential(*modules)

    def build_synth_stage(self, hidden_dims):
        modules = []
        input_channel = self.latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(input_channel, h_dim),
                nn.ReLU())
            )
            input_channel = h_dim
        self.synth_stage = nn.Sequential(*modules)

        self.synth_stage_final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.num_synth_params),
            nn.Sigmoid()
        )

    def encode(self, input: Tensor):
        """Codifica la entrada y devuelve los códigos latentes."""
        result = self.encoder(input)
        result = self.encoder_output(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def latent_to_params(self, z: Tensor):
        """Convierte los códigos latentes en los parámetros sintetizados."""
        return self.synth_stage_final_layer(self.synth_stage(z))

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

    def forward(self, input: Tensor, params: Tensor, **kwargs):
        """Propagación hacia adelante del modelo."""
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, self.latent_to_params(z), params]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        params_pred = args[4]
        params_true = args[5]
        kld_weight = kwargs['M_N']

        loss = 0

        recons_loss = F.mse_loss(recons, input, reduction='sum')
        loss += recons_loss

        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        weighted_kld_loss = self.beta * kld_weight * kld_loss
        loss += weighted_kld_loss

        return_dict = {'Reconstruction_Loss': recons_loss, 'KLD': weighted_kld_loss}

        if self.use_pink_trombone:
            # TODO: hay que quitar el hardcoding del audiolength
            regen_mel = self.pink_trombone_connection.regenerate_audio_from_pred_params(params_pred.detach().cpu().numpy(), audio_length=1.0).to(input.device)
            param_audio_regenerated_loss = F.mse_loss(input, regen_mel, reduction='sum') * self.regen_weight
            loss += param_audio_regenerated_loss
            return_dict.update({'param_audio_regenerated_loss': param_audio_regenerated_loss})

        else:
            loss_params = []
            for param_pred, param_true, beta_param in zip(params_pred.T, params_true.T, self.beta_params):
                loss_params.append(F.mse_loss(param_pred, param_true, reduction='sum') * beta_param * self.params_weight)

            loss_params_dict = {f"{param_name}_error": loss / self.params_weight for param_name, loss in zip(utils.utils.params_names, loss_params)}

            params_loss = sum(loss_params)
            loss += params_loss
            return_dict.update({'params_loss': params_loss / self.params_weight})
            return_dict.update(loss_params_dict)


        return_dict.update({'loss': loss})

        return return_dict


    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
