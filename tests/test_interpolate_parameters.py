import unittest
import numpy as np

from utils.interpolate_parameters import interpolate_params


class TestInterpolation(unittest.TestCase):
    def test_single_value_interpolation(self):
        """Testea la interpolación cuando cada parámetro tiene un único valor."""
        param_list = [[0.5], [1.0]]
        sampling_rate = 48000  # Hz
        audio_length = 1  # segundo
        expected_length = 94  # Calculado como int(48000 * 1 / 512 + 1)

        interpolated = interpolate_params(param_list, sampling_rate, audio_length)
        self.assertEqual(len(interpolated[0]), expected_length)
        self.assertTrue(np.all(interpolated[0] == 0.5))
        self.assertTrue(np.all(interpolated[1] == 1.0))

    def test_multiple_value_multi_param_interpolation(self):
        """Testea la interpolación cuando cada parámetro tiene un único valor."""
        param_list = [[0.5, 0.8], [1.0, 0.25]]
        sampling_rate = 48000  # Hz
        audio_length = 1  # segundo
        expected_length = 94  # Calculado como int(48000 * 1 / 512 + 1)

        interpolated = interpolate_params(param_list, sampling_rate, audio_length)
        self.assertEqual(len(interpolated[0]), expected_length)
        self.assertTrue(np.all(interpolated[0][0] == 0.5))
        self.assertTrue(np.all(interpolated[0][-1] == 0.8))
        self.assertTrue(np.all(interpolated[1][0] == 1.0))
        self.assertTrue(np.all(interpolated[1][-1] == 0.25))

    def test_multiple_value_interpolation(self):
        """Testea la interpolación cuando los parámetros tienen múltiples valores."""
        param_list = [[0.0, 0.5, 1.0]]
        sampling_rate = 48000  # Hz
        audio_length = 1  # segundo
        expected_length = 94  # Calculado como int(48000 * 1 / 512 + 1)

        interpolated = interpolate_params(param_list, sampling_rate, audio_length)
        self.assertEqual(len(interpolated[0]), expected_length)
        self.assertEqual(interpolated[0][0], 0.0)
        self.assertEqual(interpolated[0][-1], 1.0)

    def test_invalid_inputs(self):
        """Testea que se levanten excepciones adecuadas para entradas inválidas."""
        sampling_rate = 48000  # Ejemplo de tasa de muestreo
        # Prueba con longitud de audio cero
        with self.assertRaises(AssertionError) as context:
            interpolate_params([[0.5]], sampling_rate, 0)
        self.assertIn("Audio length cannot be zero", str(context.exception))

        # Prueba con lista de parámetros vacía
        with self.assertRaises(AssertionError) as context:
            interpolate_params([], sampling_rate, 1)
        self.assertIn("Parameter list cannot be empty", str(context.exception))


if __name__ == '__main__':
    unittest.main()
