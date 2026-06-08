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
        # run.load_model does getattr(module, name)(...), so the call lands on the
        # child mock for the named class, not on the module mock itself.
        mock_model_class.TestModel.assert_called_once_with(name='TestModel', param1='value1', param2='value2')

    @patch('run.importlib.import_module')
    def test_load_experiment_with_valid_experiment_name(self, mock_import_module):
        mock_exp_class = MagicMock()
        mock_import_module.return_value = mock_exp_class
        config = {'exp_params': {'name_exp': 'TestExperiment', 'param1': 'value1'}}
        model = MagicMock()
        load_experiment('TestExperiment', model, config)
        mock_import_module.assert_called_once_with('experiments')
        mock_exp_class.TestExperiment.assert_called_once_with(model, config['exp_params'])

    def test_load_dataloader_with_invalid_data_type(self):
        with self.assertRaises(ValueError):
            load_dataloader('InvalidDataType')

    @patch('run.importlib.import_module')
    def test_load_dataloader_with_valid_data_type(self, mock_import_module):
        mock_data_class = MagicMock()
        mock_import_module.return_value = mock_data_class
        # run.load_dataloader returns (class, json_files) via getattr without
        # instantiating, so assert on the returned class and json list instead.
        data_class, json_file = load_dataloader('spectrogram')
        mock_import_module.assert_called_once_with('data')
        self.assertIs(data_class, mock_data_class.SpectrogramDataloader)
        self.assertEqual(json_file, ['train.json', 'test.json'])

if __name__ == '__main__':
    unittest.main()