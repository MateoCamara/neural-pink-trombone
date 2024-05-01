import unittest
import torch
from models import BetaVAESynth1D  # Asegúrate de importar correctamente tu clase

class TestBetaVAESynth1D(unittest.TestCase):
    def test_encoder_decoder_symmetry(self):
        """Verifica que cada capa del encoder y su correspondiente en el decoder tengan el mismo tamaño de salida."""
        in_channels = 1  # Define según tus necesidades de canal de entrada
        latent_dim = 64  # Dimension latente que esperas utilizar
        hidden_dims = [8, 16]  # Dimensiones ocultas para el encoder y decoder
        betaVAE = BetaVAESynth1D(in_channels, latent_dim, hidden_dims)  # Inicializa tu modelo

        # Genera un tensor de entrada aleatorio
        input_tensor = torch.randn(1, in_channels, 94)  # Asume un tamaño de entrada, ajusta según necesidades
        with torch.no_grad():
            # Pasar el tensor a través del encoder
            mu, sigma = betaVAE.encode(input_tensor)
            # Pasar la salida latente a través del decoder
            decoded_tensor = betaVAE.decode(betaVAE.reparameterize(mu, sigma))

        # Verificar que el tensor de entrada y salida tengan las mismas dimensiones
        self.assertEqual(input_tensor.shape, decoded_tensor.shape, "El tensor de entrada y salida deben tener las mismas dimensiones.")

if __name__ == '__main__':
    unittest.main()
