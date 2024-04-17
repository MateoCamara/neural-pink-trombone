import unittest
import matplotlib.pyplot as plt
import librosa.display
from data.pt_data_loader_online import PTServidorDataset
import os
import json

class TestPTServidorDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = PTServidorDataset(servidor_url='127.0.0.1', servidor_port=3000, tamano_batch=1, iteraciones=1)

    def test_generate_random_audio(self):
        mel_spec, random_values = self.dataset.generate_random_audio()
        self.assertIsNotNone(mel_spec)
        self.assertIsNotNone(random_values)

    def test_normalizar_mel_spec(self):
        mel_spec, _ = self.dataset.generate_random_audio()
        normalized_mel_spec = self.dataset.normalizar_mel_spec(mel_spec)
        self.assertTrue((normalized_mel_spec >= 0).all() and (normalized_mel_spec <= 1).all())

    def test_visualize_mel_spectrogram(self):
        mel_spec, _ = self.dataset.generate_random_audio()
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec[0, :, :], x_axis='time', y_axis='mel', sr=48000, fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()

    def test_generate_records(self):
        output_dir = './test_dir'
        num_records = 10  # Use a small number for testing
        self.dataset.generate_records(output_dir, num_records)

        # Check that the correct number of audio files were created
        audio_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
        self.assertEqual(len(audio_files), num_records)

        # Check that the params.json file was created
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'params.json')))

        # Check that the params.json file contains the correct number of records
        with open(os.path.join(output_dir, 'params.json'), 'r') as f:
            params_dict = json.load(f)
        self.assertEqual(len(params_dict), num_records)

        # Clean up the generated files after the test
        for f in audio_files:
            os.remove(os.path.join(output_dir, f))
        os.remove(os.path.join(output_dir, 'params.json'))


if __name__ == '__main__':
    unittest.main()
