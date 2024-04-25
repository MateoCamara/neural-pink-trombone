import os
import unittest
import torch
from scipy.io import wavfile

from data import EmbeddingDataloader, SpectrogramDataloader


class TestEmbeddingDataloader(unittest.TestCase):

    def setUp(self):
        data_path = '../../neural-pink-trombone-data/pt_encodec_simplified'
        self.dataset = EmbeddingDataloader(os.path.join(data_path, 'train'),
                                           os.path.join(data_path, 'train.json'),
                                           audio_path='../../neural-pink-trombone-data/pt_dataset_simplified/train')

    def test_dataset_initialization_and_length(self):
        self.assertIsInstance(self.dataset, EmbeddingDataloader)
        self.assertEqual(len(self.dataset), len(self.dataset.metadata.keys()))

    def test_get_item(self):
        embedding, params, mel_spec, waveform = self.dataset[0]  # Asumiendo que no usamos la parte de audio
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertIsInstance(params, torch.Tensor)

    def test_normalization_and_denormalization(self):
        original_params = [100, 1, 20, 3, 1, 30, 1.5, 2]
        original_params = [[i] for i in original_params]
        normalized_params = self.dataset.normalizar_params(list(original_params))
        denormalized_params = self.dataset.denormalizar_params(list(normalized_params))
        for op, dp in zip(original_params, denormalized_params):
            self.assertAlmostEqual(op, dp, places=4)

    def test_mel_normalization_and_denormalization(self):
        embbeding, parameters, mel_spec, waveform = self.dataset[0]
        normalized_mel_spec = self.dataset.normalizar_mel_spec(mel_spec)
        denormalized_mel_spec = self.dataset.denormalizar_mel_spec(normalized_mel_spec)
        self.assertTrue(torch.allclose(mel_spec, denormalized_mel_spec, atol=1e-4))

    def test_mel_spectrogram_computation(self):
        fake_audio = torch.randn(1, 48000)  # Audio aleatorio de 1 segundo a 16kHz
        mel_spec = self.dataset._compute_mel_spectrogram(fake_audio, 48000, 8000)
        self.assertEqual(mel_spec.shape, (1, 128, 94))  # Confirma la forma del espectrograma Mel

    def test_audio_conversion(self):
        embedding, parameters, mel_spec_norm, waveform = self.dataset[0]  # Espectrograma Mel simulado
        mel_spec = self.dataset.denormalizar_mel_spec(mel_spec_norm)
        audio = self.dataset.convert_mel_to_audio(mel_spec)

        # save audio with scipy
        wavfile.write('test_files/test_audio_griffinlim.wav', 48000, audio)
        wavfile.write('test_files/original_test_audio.wav', 48000, waveform.numpy()[0])
        # Ojo que no pueden sonar iguales, de hecho el griffinlim debería sonar horrible, pero sufi para ver que hay info acústica ahí


class TestSpectrogramDataloader(unittest.TestCase):  # TODO: sinceramente, esto habría que quitarlo en favor del otro

    def setUp(self):
        data_path = '../../neural-pink-trombone-data/pt_dataset_simplified'
        self.dataset = SpectrogramDataloader(os.path.join(data_path, 'train'),
                                             os.path.join(data_path, 'train.json')
                                             )

    def test_dataset_initialization_and_length(self):
        self.assertIsInstance(self.dataset, SpectrogramDataloader)
        self.assertEqual(len(self.dataset), len(self.dataset.metadata.keys()))

    def test_get_item(self):
        mel_spec, params, waveform = self.dataset[0]  # Asumiendo que no usamos la parte de audio
        self.assertIsInstance(mel_spec, torch.Tensor)
        self.assertIsInstance(params, torch.Tensor)

    def test_normalization_and_denormalization(self):
        original_params = [100, 1, 20, 3, 1, 30, 1.5, 2]
        original_params = [[i] for i in original_params]
        normalized_params = self.dataset.normalizar_params(list(original_params))
        denormalized_params = self.dataset.denormalizar_params(list(normalized_params))
        for op, dp in zip(original_params, denormalized_params):
            self.assertAlmostEqual(op, dp, places=4)

    def test_mel_normalization_and_denormalization(self):
        mel_spec, parameters, waveform = self.dataset[0]
        normalized_mel_spec = self.dataset.normalizar_mel_spec(mel_spec)
        denormalized_mel_spec = self.dataset.denormalizar_mel_spec(normalized_mel_spec)
        self.assertTrue(torch.allclose(mel_spec, denormalized_mel_spec, atol=1e-4))

    def test_mel_spectrogram_computation(self):
        fake_audio = torch.randn(1, 48000)  # Audio aleatorio de 1 segundo a 16kHz
        mel_spec = self.dataset._compute_mel_spectrogram(fake_audio, 48000, 8000)
        self.assertEqual(mel_spec.shape, (1, 128, 94))  # Confirma la forma del espectrograma Mel


if __name__ == '__main__':
    unittest.main()
