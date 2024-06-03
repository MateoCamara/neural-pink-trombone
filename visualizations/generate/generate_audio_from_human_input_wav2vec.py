import importlib
import os

from visualizations.visualization_utils import set_weights_to_model, load_model

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import librosa
import numpy as np
import torch
import torchaudio
import yaml
from scipy.io.wavfile import write
from scipy.signal import savgol_filter
from tqdm import tqdm

from data import PTServidorDataset
from utils import utils
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from scipy.signal import resample



# cargar el archivo de audio

def load_audio_file(audio_path, sr):
    return librosa.load(audio_path, sr=sr)[0]


# computar su espectrograma mel

def slerp(p0, p1, t):
    """Interpola esféricamente entre dos puntos p0 y p1 usando el factor t."""
    omega = np.arccos(np.clip(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)), -1, 1))
    sin_omega = np.sin(omega)
    if sin_omega == 0:
        # Co-linear points, use linear interpolation
        return (1 - t) * p0 + t * p1
    else:
        return np.sin((1.0 - t) * omega) / sin_omega * p0 + np.sin(t * omega) / sin_omega * p1


def interpolate_embeddings(embeddings, original_fps, target_fps=94):
    """Interpola los embeddings al número de pasos por segundo objetivo."""
    times_original = np.linspace(0, 1, original_fps)
    times_target = np.linspace(0, 1, target_fps)

    # Inicializa la lista de embeddings interpolados
    interpolated_embeddings = np.zeros((target_fps, embeddings.shape[1]))

    # Calcula interpolaciones para cada paso de tiempo en el objetivo
    for i in range(target_fps):
        # Encuentra los índices de los puntos originalmente más cercanos
        idx = np.searchsorted(times_original, times_target[i]) - 1
        idx_next = min(idx + 1, original_fps - 1)

        # Calcula el factor de interpolación
        t = (times_target[i] - times_original[idx]) / (times_original[idx_next] - times_original[idx])

        # Interpola esféricamente entre los dos puntos más cercanos
        interpolated_embeddings[i] = slerp(embeddings[idx], embeddings[idx_next], t)

    return interpolated_embeddings

# Calcular su frecuencia fundamental con pyin
def compute_f0(y, sr):
    return librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

# forzar la f0 a 100 hz

def force_f0(audio, target_f0, pyin_f0, sr):
    if np.nanmean(pyin_f0) > 0:  # Evitar divisiones por cero si f0 es NaN
        stretch_factor = target_f0 / np.nanmean(pyin_f0)
    else:
        stretch_factor = 1.0

    # Cambiar la frecuencia de la señal
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=-12 * np.log2(stretch_factor))

# pasarlo por la red neuronal
def predict_parameters(model, mel_spec):
    return model.forward(input=mel_spec, params=None)

# desnormalizar los parámetros
def denormalizar_params(params):
    for i, (low, high) in enumerate(utils.bounds):
        params[i] = params[i] * (high - low) + low
    return params

# generar el audio
def generate_audio(params, size):
    servidor = PTServidorDataset(servidor_url='127.0.0.1', servidor_port=3000, tamano_batch=1, iteraciones=1,
                                number_of_changes=size)
    return servidor.generate_specific_audio(params, length= size / 94)

# guardar el audio
def save_audio(audio, output_path, sr):
    write(output_path, sr, audio)

# guardar los parámetros
def save_params(params, output_path):
    np.save(output_path, params)

if __name__ == '__main__':
    sr = 48000
    target_f0 = 100
    # get only the ones with "betaVAESynth" in the name
    all_available_configs = os.listdir('../../configs')
    all_available_configs = [config for config in all_available_configs if ("wav2vec" in config) and ("dynamic" in config)]

    human_audios_path = '../../../human_audios'
    generated_human_audios_path = '../../../generated_human_audios'

    device = torch.device('cpu')
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(device)

    for config_path in all_available_configs:
        with open(f'../../configs/{config_path}', 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        model_name = '_'.join(config_path.split('_')[1:-1])
        version_number = "version_" + config_path.split('_')[-1].split('.')[0]
        state_dict_path = os.path.join("../../logs", model_name, version_number, "checkpoints", "last.ckpt")
        model = load_model(config['model_params']['name'], config).to(device)
        model = set_weights_to_model(model, state_dict_path, device=device)

        saving_path = os.path.join(generated_human_audios_path, model_name + "_" + version_number)
        os.makedirs(saving_path, exist_ok=True)

        for audio_file_name in tqdm(os.listdir(human_audios_path)):
            audio_file = load_audio_file(os.path.join(human_audios_path, audio_file_name), sr)

            audio_sample_16k = librosa.resample(y=audio_file, orig_sr=48_000,
                                                target_sr=16000)  # necesario para wav2vec, qué mal
            audio_tensor = processor(audio_sample_16k, return_tensors="pt", padding=False,
                                     sampling_rate=16000).input_values.to(device)

            wav2vec_output = wav2vec(audio_tensor)
            embeddings = wav2vec_output.extract_features.detach().cpu().numpy()[0]

            embbeding_interpolated = torch.tensor(interpolate_embeddings(embeddings, original_fps=embeddings.shape[0], target_fps=int(np.round(audio_file.shape[0]/512)))).float()


            pyin_f0 = compute_f0(audio_file, sr)[0]
            pyin_f0 = np.nan_to_num(pyin_f0, nan=np.nanmean(pyin_f0))
            pyin_f0 = resample(pyin_f0, len(embbeding_interpolated))

            pred_params = []
            for index, (mel_time_instant, pyin_time_instant) in tqdm(enumerate(zip(embbeding_interpolated, pyin_f0))):
                # input es igual al instante actual y al anterior
                if index == 0:
                    input = torch.cat([embbeding_interpolated[index, :].unsqueeze(0)] * 2, dim=0)
                else:
                    input = embbeding_interpolated[index-1:index+1, :]
                if input.shape[0] == 1:
                    raise ValueError('Input shape is wrong')

                input = input.unsqueeze(0)
                pred_params.append(predict_parameters(model, input)[0].detach().cpu().numpy()[0])

            denorm_params = []
            for params, pyin in zip(pred_params, pyin_f0):
                aux_denorm_params = denormalizar_params(params).tolist()
                # add at the first position the f0
                aux_denorm_params.insert(0, 1) # voiceness
                aux_denorm_params.insert(0, pyin)

                denorm_params.append(aux_denorm_params)
            size = len(denorm_params)
            denorm_params = np.array(denorm_params).T.tolist()

            filtered_params = []
            for i in range(len(denorm_params)):
                filtered_params.append(savgol_filter(denorm_params[i], window_length=32, polyorder=2))

            audio = generate_audio(denorm_params, size)
            audio_filtered = generate_audio(filtered_params, size)

            save_audio(audio, os.path.join(saving_path, audio_file_name), sr)
            save_audio(audio_filtered, os.path.join(saving_path, audio_file_name[:-4] + '_filtered.wav'), sr)

            save_params(denorm_params, os.path.join(saving_path, audio_file_name[:-4] + '.npy'))
            save_params(filtered_params, os.path.join(saving_path, audio_file_name[:-4] + '_filtered.npy'))

