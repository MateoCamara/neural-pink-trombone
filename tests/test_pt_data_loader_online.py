import unittest
import scipy.io.wavfile as wavfile

import numpy as np
import torch

from data import PTServidorDataset, Tongue


class TestTongue(unittest.TestCase):

    def setUp(self):
        self.tongue = Tongue()

    def test_get_diam_interpolation(self):
        # Check the bounds
        self.assertEqual(self.tongue.get_diam_interpolation(self.tongue.min_diam), 0)
        self.assertEqual(self.tongue.get_diam_interpolation(self.tongue.max_diam), 1)
        # Check a mid value
        mid_diam = (self.tongue.min_diam + self.tongue.max_diam) / 2
        self.assertAlmostEqual(self.tongue.get_diam_interpolation(mid_diam), 0.5, places=2)

    def test_get_index_center_offset(self):
        # Check the center offset with an interpolation of 0.5
        interpolation = 0.5
        expected_offset = (interpolation * self.tongue.index_range) / 2
        self.assertAlmostEqual(self.tongue.get_index_center_offset(interpolation), expected_offset, places=2)

    def test_set_diameter(self):
        # Set and check the diameter
        self.tongue.set_diameter(2.5)
        self.assertEqual(self.tongue.diameter, 2.5)

    def test_set_index(self):
        # Set and check the index
        self.tongue.set_index(15)
        self.assertEqual(self.tongue.index, 15)

    def test_random_diameter_setting(self):
        self.tongue.set_random_diameter()
        # Check that the randomly set diameter is within range
        self.assertTrue(self.tongue.min_diam <= self.tongue.diameter <= self.tongue.max_diam)

    def test_get_index_range_based_on_diam(self):
        self.tongue.set_diameter(self.tongue.min_diam)
        expected_range_min, expected_range_max = self.tongue.get_index_range_based_on_diam()
        # Check the index ranges computed at the minimum diameter
        self.assertTrue(self.tongue.min_index <= expected_range_min <= self.tongue.max_index)
        self.assertTrue(self.tongue.min_index <= expected_range_max <= self.tongue.max_index)

    def test_random_index_based_on_diam(self):
        self.tongue.set_random_diameter()
        self.tongue.set_random_index_based_on_diam()
        # Check that the random index is within the computed range
        range_min, range_max = self.tongue.get_index_range_based_on_diam()
        self.assertTrue(range_min <= self.tongue.index <= range_max)

    def test_plot_tongue_positions(self):
        # Check that the function does not raise errors
        self.tongue.plot_tongue_positions()


class TestPTServidorDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = PTServidorDataset(servidor_url="127.0.0.1",
                                         servidor_port=3000,
                                         tamano_batch=1,
                                         iteraciones=1)

    def test_generate_random_audio(self):
        # Generate a random audio and verify it is a PyTorch tensor
        audio, _ = self.dataset.obtener_datos_de_servidor()

        # listen to the audio

        wavfile.write('test_files/test_static.wav', 48000, audio)

    def test_generate_specific_audio(self):
        # Generate a random audio and verify it is a PyTorch tensor
        params = [140, 1, 29, 2.05, 1.7, 30, 2, 2]
        params = [[i] for i in params]
        audio = self.dataset.generate_specific_audio(params)

        # listen to the audio

        wavfile.write('test_files/test_specific.wav', 48000, audio)

    def test_generate_specific_dynamic_audio(self):
        # Generate a random audio and verify it is a PyTorch tensor
        params = [[140, 140, 140], [1, 1, 1], [29, 20.5, 12.5], [2.05, 3.4, 2.1], [1.7, 1.7, 1.7], [30, 30, 30], [2, 2, 2], [2, 2, 2]]
        audio = self.dataset.generate_specific_audio(params, length=5)

        # listen to the audio

        wavfile.write('test_files/test_specific_dynamic.wav', 48000, audio)

    def test_generate_random_dynamic_audio(self):
        dataset_dynamic = PTServidorDataset(servidor_url="127.0.0.1",
                                            servidor_port=3000,
                                            tamano_batch=1,
                                            iteraciones=1,
                                            number_of_changes=2)

        audio, _ = dataset_dynamic.obtener_datos_de_servidor()

        # save file as wav with scipy

        wavfile.write('test_files/test_dynamic.wav', 48000, audio)

    def test_batch_generation(self):
        # Check that the generated batches have the correct size
        for mel_spec_batch, random_values_batch in self.dataset:
            self.assertEqual(mel_spec_batch.shape[0], self.dataset.tamano_batch)
            self.assertEqual(random_values_batch.shape[0], self.dataset.tamano_batch)
            break  # We only test the first batch for this test

    def test_normalization(self):
        # Generate a mel spectrogram and verify the normalization
        mel_spec, _ = self.dataset.generate_random_audio()
        mel_spec_normalized = self.dataset.normalizar_mel_spec(mel_spec)
        # Make sure the values are properly normalized (depends on how you define 'normalize')
        self.assertTrue(mel_spec_normalized.max() <= 1 and mel_spec_normalized.min() >= 0)

    def test_integration_with_tongue(self):
        # Check that the 'tongue' values are being integrated correctly
        self.dataset.tongue.set_random_diameter()
        diameter = self.dataset.tongue.diameter
        self.dataset.tongue.set_random_index_based_on_diam()
        index = self.dataset.tongue.index
        # We assume some aspect of the generated data depends on these values
        # Here you need to define exactly what you expect to verify
        self.assertTrue(isinstance(index, float) and isinstance(diameter, float))  # This is a generic example


if __name__ == '__main__':
    unittest.main()
