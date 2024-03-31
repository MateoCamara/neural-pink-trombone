import unittest
import matplotlib.pyplot as plt
import librosa.display
from data.pt_data_loader import PTServidorDataset


class TestPTServidorDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = PTServidorDataset(servidor_url='127.0.0.1', servidor_port=3000, tamano_batch=1, iteraciones=1)

    def test_obtener_datos_de_servidor(self):
        mel_spec, random_values = self.dataset.obtener_datos_de_servidor()
        self.assertIsNotNone(mel_spec)
        self.assertIsNotNone(random_values)

    def test_normalizar_mel_spec(self):
        mel_spec, _ = self.dataset.obtener_datos_de_servidor()
        normalized_mel_spec = self.dataset.normalizar_mel_spec(mel_spec)
        self.assertTrue((normalized_mel_spec >= 0).all() and (normalized_mel_spec <= 1).all())

    def test_visualize_mel_spectrogram(self):
        mel_spec, _ = self.dataset.obtener_datos_de_servidor()
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec[0, :, :], x_axis='time', y_axis='mel', sr=48000, fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
