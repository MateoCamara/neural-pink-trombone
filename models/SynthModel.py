import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar
import lightning as L

import utils.utils
from models.base_pink_trombone_connection import PinkTromboneConnection

Tensor = TypeVar('torch.tensor')


class SynthStage(L.LightningModule):
    num_iter = 0  # Variable estática global para llevar la cuenta de las iteraciones

    def __init__(self, codec_dim: int, time_dim: int, hidden_dims: List = None, output_dims: int = None,
                 beta_params: list = [], params_weight: int = 1, **kwargs):
        super().__init__()

        self.codec_dim = codec_dim
        self.time_dim = time_dim
        self.beta_params = beta_params
        self.params_weight = params_weight

        if hidden_dims is None:
            hidden_dims = [codec_dim, codec_dim // 2, codec_dim // 4, codec_dim // 8]

        if output_dims is None:
            output_dims = 8

        if beta_params is None:
            self.beta_params = [1] * output_dims

        self.build_stage(hidden_dims=hidden_dims, output_dims=output_dims)

        self.use_pink_trombone = kwargs.get('use_pink_trombone', False)

        if self.use_pink_trombone:
            self.regen_weight = kwargs.get('regen_weight', 1.0)
            self.pink_trombone_connection = PinkTromboneConnection(kwargs.get('pt_server', '127.0.0.1'), kwargs.get('pt_port', 3000))


    def build_stage(self, hidden_dims, output_dims):
        modules = []
        input_channel = 1
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(input_channel, h_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(h_dim) if h_dim != hidden_dims[0] else nn.Identity()
            ))
            input_channel = h_dim
        self.synth_stage = nn.Sequential(*modules)

        flat_size = self._calculate_output_size(self.synth_stage,
                                                torch.randn(1, 1, self.codec_dim, self.time_dim)).numel()

        self.synth_stage_final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, flat_size // 2),
            nn.ReLU(),
            nn.Linear(flat_size // 2, 8),
            nn.Sigmoid()
        )

    def forward(self, input: Tensor, params: Tensor, original_audio: Tensor = None):
        x = self.synth_stage(input)
        result = self.synth_stage_final_layer(x)
        return [result, input, params, original_audio]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        params_pred = args[0]
        input = args[1]
        params_true = args[2]
        original_audio = args[3]

        loss = 0
        return_dict = {}

        if self.use_pink_trombone:
            # TODO: hay que quitar el hardcoding del audiolength
            regen_mel = self.pink_trombone_connection.regenerate_audio_from_pred_params(params_pred.detach().cpu().numpy(), audio_length=1.0).to(input.device)
            param_audio_regenerated_loss = F.mse_loss(original_audio, regen_mel, reduction='sum') * self.regen_weight
            loss += param_audio_regenerated_loss
            return_dict.update({'param_audio_regenerated_loss': param_audio_regenerated_loss})

        else:
            loss_params = []
            for param_pred, param_true, self.beta_param in zip(params_pred, params_true, self.beta_params):
                loss_params.append(F.mse_loss(param_pred, param_true, reduction='sum') * self.beta_param * self.params_weight)

            loss_params_dict = {f"{param_name}_error": loss for param_name, loss in zip(utils.utils.params_names, loss_params)}

            params_loss = sum(loss_params)
            loss += params_loss
            return_dict.update({'params_loss': params_loss})
            return_dict.update(loss_params_dict)

        return_dict = {'loss': loss}

        return return_dict

    def _calculate_output_size(self, model, input_tensor):
        """Calcula el tamaño de salida de un modelo dado un tensor de entrada."""
        with torch.no_grad():
            for module in model:
                input_tensor = module(input_tensor)
        return input_tensor.size()

    def print_model_summary(self):
        """
        Prints a summary of the model layers and parameters.
        """
        print("Model Summary:\n")
        total_params = 0
        for name, module in self.model.named_modules():
            # Ignoring modules that do not have learnable parameters
            if list(module.parameters(recurse=False)):
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"{name} - {module.__class__.__name__} : {num_params} trainable parameters")
                total_params += num_params
        print(f"\nTotal trainable parameters: {total_params}")
