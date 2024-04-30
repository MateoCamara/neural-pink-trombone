import unittest
from unittest.mock import patch, MagicMock
from run import load_model, load_experiment, load_dataloader

class TestRunMethods(unittest.TestCase):

    @patch('run.importlib.import_module')
    def test_load_model_with_valid_model_name(self, mock_import_module):
        mock_model_class = MagicMock()
        mock_import_module.return_value = mock_model_class
        config = {'model_params': {'name': 'TestModel', 'param1': 'value1'}, 'exp_params': {'param2': 'value2'}}
        load_model('TestModel', config)
        mock_import_module.assert_called_once_with('models')
        mock_model_class.assert_called_once_with(name='TestModel', param1='value1', param2='value2')

    @patch('run.importlib.import_module')
    def test_load_experiment_with_valid_experiment_name(self, mock_import_module):
        mock_exp_class = MagicMock()
        mock_import_module.return_value = mock_exp_class
        config = {'exp_params': {'name_exp': 'TestExperiment', 'param1': 'value1'}}
        model = MagicMock()
        load_experiment('TestExperiment', model, config)
        mock_import_module.assert_called_once_with('experiments')
        mock_exp_class.assert_called_once_with(model, config['exp_params'])

    def test_load_dataloader_with_invalid_data_type(self):
        with self.assertRaises(ValueError):
            load_dataloader('InvalidDataType')

    @patch('run.importlib.import_module')
    def test_load_dataloader_with_valid_data_type(self, mock_import_module):
        mock_data_class = MagicMock()
        mock_import_module.return_value = mock_data_class
        load_dataloader('spectrogram')
        mock_import_module.assert_called_once_with('data')
        mock_data_class.assert_called_once_with()

if __name__ == '__main__':
    unittest.main()