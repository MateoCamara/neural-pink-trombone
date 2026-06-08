import unittest
import numpy as np

from utils.interpolate_parameters import interpolate_params


class TestInterpolation(unittest.TestCase):
    def test_single_value_interpolation(self):
        """Test interpolation when each parameter has a single value."""
        param_list = [[0.5], [1.0]]
        sampling_rate = 48000  # Hz
        audio_length = 1  # second
        expected_length = 94  # Computed as int(48000 * 1 / 512 + 1)

        interpolated = interpolate_params(param_list, sampling_rate, audio_length)
        self.assertEqual(len(interpolated[0]), expected_length)
        self.assertTrue(np.all(interpolated[0] == 0.5))
        self.assertTrue(np.all(interpolated[1] == 1.0))

    def test_multiple_value_multi_param_interpolation(self):
        """Test interpolation with several parameters, each having two values."""
        param_list = [[140, 140], [1, 1], [20, 27.5], [3.4, 2.2], [1.7, 1.7], [30, 30], [2, 2], [2, 2]]
        sampling_rate = 48000  # Hz
        audio_length = 1  # second
        expected_length = 94  # Computed as int(48000 * 1 / 512 + 1)

        interpolated = interpolate_params(param_list, sampling_rate, audio_length)
        self.assertEqual(len(interpolated[0]), expected_length)
        self.assertTrue(np.all(interpolated[0][0] == 0.5))
        self.assertTrue(np.all(interpolated[0][-1] == 0.8))
        self.assertTrue(np.all(interpolated[1][0] == 1.0))
        self.assertTrue(np.all(interpolated[1][-1] == 0.25))

    def test_multiple_value_interpolation(self):
        """Test interpolation when parameters have multiple values."""
        param_list = [[0.0, 0.5, 1.0]]
        sampling_rate = 48000  # Hz
        audio_length = 1  # second
        expected_length = 94  # Computed as int(48000 * 1 / 512 + 1)

        interpolated = interpolate_params(param_list, sampling_rate, audio_length)
        self.assertEqual(len(interpolated[0]), expected_length)
        self.assertEqual(interpolated[0][0], 0.0)
        self.assertEqual(interpolated[0][-1], 1.0)

    def test_invalid_inputs(self):
        """Test that appropriate exceptions are raised for invalid inputs."""
        sampling_rate = 48000  # Example sampling rate
        # Test with zero audio length
        with self.assertRaises(AssertionError) as context:
            interpolate_params([[0.5]], sampling_rate, 0)
        self.assertIn("Audio length cannot be zero", str(context.exception))

        # Test with an empty parameter list
        with self.assertRaises(AssertionError) as context:
            interpolate_params([], sampling_rate, 1)
        self.assertIn("Parameter list cannot be empty", str(context.exception))


if __name__ == '__main__':
    unittest.main()
