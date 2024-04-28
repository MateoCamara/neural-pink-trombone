import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar
from models.base import BaseMLP

Tensor = TypeVar('torch.tensor')

# MLP designed to be fed the latent parameter vector + cosine distance of the latter w.r.t. original parameter vector
# and return an estimation of the re-synthesis loss (MSE) in the time-frequency domain (we bypass full re-synthesis).
class param2loss(BaseMLP):
    num_iter = 0  # Variable estática global para llevar la cuenta de las iteraciones

    def __init__(self, in_channels: int = 6 + 1, out_channels: int = 1, hidden_dims: List = None, **kwargs):
        super(param2loss, self).__init__()
        # Definición de variables internas
        self.out_channels = out_channels
        # Inicialización de dimensiones ocultas si no se proporcionan
        if hidden_dims is None:
            hidden_dims = [32, 16, 8, 4]
        # Construye red MLP
        self.build_network(in_channels, out_channels, hidden_dims)
        # Print neural graph
        self.print_neural_graph()

    def build_network(self, in_channels, out_channels, hidden_dims):
        """Construye la red MLP"""
        # Input and hidden layers
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, out_channels=h_dim),
                nn.ReLU(),
                # Batchnorm, dropout...
                ))
            in_channels = h_dim
        # Output layer
        modules.append(nn.Sequential(
                nn.Linear(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(), # Final ReLU activation: output is nonnegatve (loss proxy)
                ))

        self.network = nn.Sequential(*modules)


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
        return self.network(input)
    
    def loss_function(self, *args, **kwargs) -> dict:
        # Arguments
        self.num_iter += 1
        estimated_loss      = args[0]
        synthesised_output  = args[1]
        synthesised_input   = args[2]
        # Compute loss
        recons_loss = F.mse_loss(synthesised_output, synthesised_input, reduction='sum')
        loss = (estimated_loss - recons_loss) ** 2

        return {'loss': loss}
