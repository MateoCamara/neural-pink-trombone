import os
import unittest

import librosa
import torch
from scipy.io import wavfile

from data import EmbeddingDataloader, SpectrogramDataloader, DynamicEmbeddingDataloader


class TestEmbeddingDataloaderWav2vec(unittest.TestCase):

    def setUp(self):
        data_path = '../../neural-pink-trombone-data/pt_wav2vec_simplified'
        self.dataset = EmbeddingDataloader(os.path.join(data_path, 'train'),
                                           os.path.join(data_path, 'train.json'),
                                           audio_path='../../neural-pink-trombone-data/pt_dataset_simplified/train')

    def test_dataset_initialization_and_length(self):
        self.assertIsInstance(self.dataset, EmbeddingDataloader)
        self.assertEqual(len(self.dataset), len(self.dataset.metadata.keys()))

    def test_get_item(self):
        embedding, params, mel_spec, waveform = self.dataset[0]  # Assuming we do not use the audio part
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
        fake_audio = torch.randn(1, 48000)  # Random 1-second audio at 16kHz
        mel_spec = self.dataset._compute_mel_spectrogram(fake_audio, 48000, 8000)
        self.assertEqual(mel_spec.shape, (1, 128, 94))  # Confirm the shape of the mel spectrogram

    def test_audio_conversion(self):
        embedding, parameters, mel_spec_norm, waveform = self.dataset[0]  # Simulated mel spectrogram
        mel_spec = self.dataset.denormalizar_mel_spec(mel_spec_norm)
        audio = self.dataset.convert_mel_to_audio(mel_spec)

        # save audio with scipy
        wavfile.write('test_files/test_audio_griffinlim.wav', 48000, audio)
        wavfile.write('test_files/original_test_audio.wav', 48000, waveform.numpy()[0])
        # Note they won't sound the same; Griffin-Lim should sound terrible, but enough to confirm there is acoustic info there

    def test_decoding(self):
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2FeatureExtractor

        embedding, parameters, mel_spec_norm, waveform = self.dataset[0]

        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        tokenizer = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

        audio_sample_16k = librosa.resample(y=waveform.numpy(), orig_sr=48_000,
    target_sr=16000)

        input_values = processor(audio_sample_16k, return_tensors="pt", padding=False).input_values

        # Suppose 'logits' is the model output after passing your audio through it
        output = model(input_values)  # make sure to get the logits from the model
        # logits = output.logits
        embeddings = output.extract_features
        # Decode the logits to text
        # predicted_ids = torch.argmax(logits, dim=-1)
        # transcription = tokenizer.batch_decode(predicted_ids)

        # Print the decoded results
        print("Decoded Text:", embeddings)


class TestEmbeddingDataloaderEncodec(unittest.TestCase):

    def setUp(self):
        data_path = '../../neural-pink-trombone-data/pt_encodec_simplified'
        self.dataset = EmbeddingDataloader(os.path.join(data_path, 'train'),
                                           os.path.join(data_path, 'train.json'),
                                           audio_path='../../neural-pink-trombone-data/pt_dataset_simplified/train')

    def test_dataset_initialization_and_length(self):
        self.assertIsInstance(self.dataset, EmbeddingDataloader)
        self.assertEqual(len(self.dataset), len(self.dataset.metadata.keys()))

    def test_get_item(self):
        embedding, params, mel_spec, waveform = self.dataset[0]  # Assuming we do not use the audio part
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
        fake_audio = torch.randn(1, 48000)  # Random 1-second audio at 16kHz
        mel_spec = self.dataset._compute_mel_spectrogram(fake_audio, 48000, 8000)
        self.assertEqual(mel_spec.shape, (1, 128, 94))  # Confirm the shape of the mel spectrogram

    def test_audio_conversion(self):
        embedding, parameters, mel_spec_norm, waveform = self.dataset[0]  # Simulated mel spectrogram
        mel_spec = self.dataset.denormalizar_mel_spec(mel_spec_norm)
        audio = self.dataset.convert_mel_to_audio(mel_spec)

        # save audio with scipy
        wavfile.write('test_files/test_audio_griffinlim.wav', 48000, audio)
        wavfile.write('test_files/original_test_audio.wav', 48000, waveform.numpy()[0])
        # Note they won't sound the same; Griffin-Lim should sound terrible, but enough to confirm there is acoustic info there

    def test_decoding(self):
        from encodec import EncodecModel

        embedding, parameters, mel_spec_norm, waveform = self.dataset[0]

        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(12.0)
        model.eval()
        model.to('cpu')

        audio_tensor = waveform.float().unsqueeze(0)  # Add batch dimension
        embeddings = model.encode(audio_tensor)
        embeddings = embeddings[0][0]
        # Inverse process to get audio from embeddings
        arr = embeddings.to('cpu')
        emb = model.quantizer.decode(arr.transpose(0, 1))

        out = model.decoder(emb)

        audio_arr = out.detach().cpu().numpy().squeeze()

        # save audio with scipy
        wavfile.write('test_files/test_audio_encodec.wav', 48000, audio_arr)


class TestDynamicEmbeddingDataloader(unittest.TestCase):  # TODO: honestly, this should be removed in favor of the other one

    def setUp(self):
        data_path = '../../neural-pink-trombone-data/pt_encodec_dynamic_simplified'
        self.dataset = DynamicEmbeddingDataloader(os.path.join(data_path, 'train'),
                                             os.path.join(data_path, 'train_interpolated.json')
                                             )

    def test_dataset_initialization_and_length(self):
        self.assertIsInstance(self.dataset, DynamicEmbeddingDataloader)
        self.assertEqual(len(self.dataset), len(self.dataset.metadata.keys()))

    def test_get_item(self):
        embbeding_interpolated, parameters, previous_parameters = self.dataset[0]  # Assuming we do not use the audio part
        self.assertIsInstance(embbeding_interpolated, torch.Tensor)
        self.assertIsInstance(parameters, torch.Tensor)
        self.assertIsInstance(previous_parameters, torch.Tensor)


class TestSpectrogramDataloader(unittest.TestCase):  # TODO: honestly, this should be removed in favor of the other one

    def setUp(self):
        data_path = '../../neural-pink-trombone-data/pt_dataset_simplified'
        self.dataset = SpectrogramDataloader(os.path.join(data_path, 'train'),
                                             os.path.join(data_path, 'train.json')
                                             )

    def test_dataset_initialization_and_length(self):
        self.assertIsInstance(self.dataset, SpectrogramDataloader)
        self.assertEqual(len(self.dataset), len(self.dataset.metadata.keys()))

    def test_get_item(self):
        mel_spec, params, waveform = self.dataset[0]  # Assuming we do not use the audio part
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
        fake_audio = torch.randn(1, 48000)  # Random 1-second audio at 16kHz
        mel_spec = self.dataset._compute_mel_spectrogram(fake_audio, 48000, 8000)
        self.assertEqual(mel_spec.shape, (1, 128, 94))  # Confirm the shape of the mel spectrogram


if __name__ == '__main__':
    unittest.main()
