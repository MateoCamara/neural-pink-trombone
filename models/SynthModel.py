import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar
import lightning as L

Tensor = TypeVar('torch.tensor')


class SynthStage(L.LightningModule):
    num_iter = 0  # Variable estática global para llevar la cuenta de las iteraciones

    def __init__(self, codec_dim: int, time_dim: int, hidden_dims: List = None, output_dims: int = None, **kwargs):
        super().__init__()

        self.codec_dim = codec_dim
        self.time_dim = time_dim

        if hidden_dims is None:
            hidden_dims = [codec_dim, codec_dim // 2, codec_dim // 4, codec_dim // 8]

        if output_dims is None:
            output_dims = 8

        self.build_stage(hidden_dims=hidden_dims, output_dims=output_dims)

    def build_stage(self, hidden_dims, output_dims):
        modules = []
        input_channel = 1
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(input_channel, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim) if h_dim != hidden_dims[0] else nn.ReLU(),
                nn.ReLU())
            )
            input_channel = h_dim
        self.synth_stage = nn.Sequential(*modules)

        flat_size = self._calculate_output_size(self.synth_stage, torch.randn(1, 1, self.codec_dim, self.time_dim)).numel()

        self.synth_stage_final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, flat_size // 2),
            nn.ReLU(),
            nn.Linear(flat_size // 2, 8),
            nn.Sigmoid()
        )

    def forward(self, input: Tensor, params: Tensor):
        x = self.synth_stage(input)
        result = self.synth_stage_final_layer(x)
        return [result, input, params]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        params_pred = args[0]
        input = args[1]
        params_true = args[2]

        loss = F.mse_loss(params_pred, params_true, reduction='sum')

        return {'loss': loss}

    def _calculate_output_size(self, model, input_tensor):
        """Calcula el tamaño de salida de un modelo dado un tensor de entrada."""
        with torch.no_grad():
            for module in model:
                input_tensor = module(input_tensor)
        return input_tensor.size()
