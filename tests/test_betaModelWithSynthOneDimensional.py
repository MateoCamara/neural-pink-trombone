import unittest
import torch
from models import BetaVAESynth1D  # make sure to import the class correctly

class TestBetaVAESynth1D(unittest.TestCase):
    def test_encoder_decoder_symmetry(self):
        """Check that each encoder layer and its decoder counterpart have the same output size."""
        in_channels = 1  # Define according to your input-channel needs
        latent_dim = 64  # Latent dimension you expect to use
        hidden_dims = [8, 16]  # Hidden dimensions for the encoder and decoder
        betaVAE = BetaVAESynth1D(in_channels, latent_dim, hidden_dims)  # Initialize the model

        # Generate a random input tensor
        input_tensor = torch.randn(1, in_channels, 94)  # Assumes an input size; adjust as needed
        with torch.no_grad():
            # Pass the tensor through the encoder
            mu, sigma = betaVAE.encode(input_tensor)
            # Pass the latent output through the decoder
            decoded_tensor = betaVAE.decode(betaVAE.reparameterize(mu, sigma))

        # Check that the input and output tensors have the same dimensions
        self.assertEqual(input_tensor.shape, decoded_tensor.shape, "The input and output tensors must have the same dimensions.")

if __name__ == '__main__':
    unittest.main()
