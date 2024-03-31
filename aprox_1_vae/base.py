from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
from abc import abstractmethod
import lightning as L


Tensor = TypeVar('torch.tensor')

class BaseVAE(L.LightningModule):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    # @abstractmethod
    # def training_step(self, batch, batch_idx):
    #     dataset = MiServidorIterableDataset("http://mi.servidor.com", batch_size=32)
    #     # Aquí el batch_size se configura como 1 porque el batching ya es manejado por el IterableDataset.
    #     loader = DataLoader(dataset, batch_size=1,
    #                         num_workers=0)  # num_workers=0 para evitar problemas con IterableDataset.
    #     return loader
    #
    # @abstractmethod
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=1e-3)
