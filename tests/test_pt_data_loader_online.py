import unittest

import torch

from data import PTServidorDataset, Tongue

class TestTongue(unittest.TestCase):

    def setUp(self):
        self.tongue = Tongue()

    def test_get_diam_interpolation(self):
        # Comprobación de los límites
        self.assertEqual(self.tongue.get_diam_interpolation(self.tongue.min_diam), 0)
        self.assertEqual(self.tongue.get_diam_interpolation(self.tongue.max_diam), 1)
        # Comprobación de un valor medio
        mid_diam = (self.tongue.min_diam + self.tongue.max_diam) / 2
        self.assertAlmostEqual(self.tongue.get_diam_interpolation(mid_diam), 0.5, places=2)

    def test_get_index_center_offset(self):
        # Comprobación del offset del centro con interpolación de 0.5
        interpolation = 0.5
        expected_offset = (interpolation * self.tongue.index_range) / 2
        self.assertAlmostEqual(self.tongue.get_index_center_offset(interpolation), expected_offset, places=2)

    def test_set_diameter(self):
        # Establecer y comprobar el diámetro
        self.tongue.set_diameter(2.5)
        self.assertEqual(self.tongue.diameter, 2.5)

    def test_set_index(self):
        # Establecer y comprobar el índice
        self.tongue.set_index(15)
        self.assertEqual(self.tongue.index, 15)

    def test_random_diameter_setting(self):
        self.tongue.set_random_diameter()
        # Verificar que el diámetro establecido aleatoriamente está dentro del rango
        self.assertTrue(self.tongue.min_diam <= self.tongue.diameter <= self.tongue.max_diam)

    def test_get_index_range_based_on_diam(self):
        self.tongue.set_diameter(self.tongue.min_diam)
        expected_range_min, expected_range_max = self.tongue.get_index_range_based_on_diam()
        # Verificar los rangos de índices calculados en el mínimo diámetro
        self.assertTrue(self.tongue.min_index <= expected_range_min <= self.tongue.max_index)
        self.assertTrue(self.tongue.min_index <= expected_range_max <= self.tongue.max_index)

    def test_random_index_based_on_diam(self):
        self.tongue.set_random_diameter()
        self.tongue.set_random_index_based_on_diam()
        # Verificar que el índice aleatorio está dentro del rango calculado
        range_min, range_max = self.tongue.get_index_range_based_on_diam()
        self.assertTrue(range_min <= self.tongue.index <= range_max)

    def test_plot_tongue_positions(self):
        # Verificar que la función no arroja errores
        self.tongue.plot_tongue_positions()

class TestPTServidorDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = PTServidorDataset(servidor_url="127.0.0.1",
                                         servidor_port=3000,
                                         tamano_batch=1,
                                         iteraciones=1)

    def test_batch_generation(self):
        # Comprueba que los lotes generados tienen el tamaño correcto
        for mel_spec_batch, random_values_batch in self.dataset:
            self.assertEqual(mel_spec_batch.shape[0], self.dataset.tamano_batch)
            self.assertEqual(random_values_batch.shape[0], self.dataset.tamano_batch)
            break  # Solo probamos el primer lote para esta prueba

    def test_normalization(self):
        # Genera un espectrograma de Mel y verifica la normalización
        mel_spec, _ = self.dataset.generate_random_audio()
        mel_spec_normalized = self.dataset.normalizar_mel_spec(mel_spec)
        # Asegurarse de que los valores estén normalizados adecuadamente (esto depende de cómo definas 'normalizar')
        self.assertTrue(mel_spec_normalized.max() <= 1 and mel_spec_normalized.min() >= 0)

    def test_integration_with_tongue(self):
        # Verificar que los valores de 'tongue' están siendo integrados correctamente
        self.dataset.tongue.set_random_diameter()
        diameter = self.dataset.tongue.diameter
        self.dataset.tongue.set_random_index_based_on_diam()
        index = self.dataset.tongue.index
        # Suponemos que algún aspecto de los datos generados depende de estos valores
        # Aquí necesitas definir exactamente qué esperas verificar
        self.assertTrue(isinstance(index, float) and isinstance(diameter, float))  # Este es un ejemplo genérico


if __name__ == '__main__':
    unittest.main()
