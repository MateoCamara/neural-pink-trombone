import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keyboard

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

import sounddevice as sd



# load the audio file

def load_audio_file(audio_path, sr):
    return librosa.load(audio_path, sr=sr)[0]

def load_model(config_path, model_checkpoint_path):
    def _load_model(model_name, config):
        models_module = importlib.import_module("models")
        model_class = getattr(models_module, model_name)
        return model_class(**config['model_params'], **config['exp_params'])

    with open(f'../configs/{config_path}', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    state_dict_path = os.path.join(model_checkpoint_path)
    model = _load_model(config['model_params']['name'], config)
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))['state_dict']
    fixed_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(fixed_state_dict)
    return model

# compute its mel spectrogram

def compute_mel_spectrogram(audio, sr, power=True):
    """
    Compute the MEL spectrogram of a given audio signal.
    """
    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        f_min=0,
        f_max=8000,
        n_mels=128,
        window_fn=torch.hann_window,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode='reflect',
        mel_scale='htk'  # Ensure mel scale is 'htk' for compatibility
    )

    S = spec_transform(torch.Tensor(audio))

    if power:
        db_transform = torchaudio.transforms.AmplitudeToDB('power', top_db=80.)
        S_dB = db_transform(S)
        return S_dB
    else:
        return S

# normalize the mel spectrogram

def normalizar_mel_spec(mel_spec):
    min_spec_value = -65
    max_spec_value = 45
    mel_spec = np.clip(mel_spec, min_spec_value, max_spec_value)
    mel_spec = (mel_spec - min_spec_value) / (max_spec_value - min_spec_value)
    return mel_spec

# Compute its fundamental frequency with pyin
def compute_f0(y, sr):
    return librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

# force the f0 to 100 Hz

def force_f0(audio, target_f0, pyin_f0, sr):
    if np.nanmean(pyin_f0) > 0:  # Avoid division by zero if f0 is NaN
        stretch_factor = target_f0 / np.nanmean(pyin_f0)
    else:
        stretch_factor = 1.0

    # Change the frequency of the signal
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=-12 * np.log2(stretch_factor))

# pass it through the neural network
def predict_parameters(model, mel_spec):
    return model.forward(input=mel_spec, params=None)

# denormalize the parameters
def denormalizar_params(params):
    for i, (low, high) in enumerate(utils.bounds):
        params[i] = params[i] * (high - low) + low
    return params

# generate the audio
def generate_audio(params, size):
    servidor = PTServidorDataset(servidor_url='127.0.0.1', servidor_port=3000, tamano_batch=1, iteraciones=1,
                                number_of_changes=size)
    return servidor.generate_specific_audio(params, length= size / 94)

# save the audio
def save_audio(audio, output_path, sr):
    write(output_path, sr, audio)

# save the parameters
def save_params(params, output_path):
    np.save(output_path, params)


def record_audio(fs):
    """Record audio from the microphone while the spacebar is held down,
    starting to record when significant sound is detected and trimming the leading silence."""
    print("Press the spacebar to start monitoring the sound...")
    audio_data = []
    threshold = 0  # Threshold for sound detection (adjust as needed)

    def callback(indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > threshold:
            audio_data.extend(indata.copy())  # Save audio data when it exceeds the threshold

    # Set up the stream to monitor and record
    keyboard.wait('space')  # Wait for the spacebar to be pressed
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        print("Starting sound monitoring...")
        while keyboard.is_pressed('space'):  # While the spacebar is held down
            sd.sleep(100)
        print("Recording paused.")

    # Concatenate all recorded audio fragments
    if audio_data:
        audio_data = np.concatenate(audio_data, axis=0)

    audio_data = librosa.util.normalize(audio_data)
    return audio_data

def play_audio(audio_data, fs):
    """Play back the recorded audio."""
    if audio_data is not None:
        print("Playing back the recorded audio...")
        sd.play(audio_data, samplerate=fs)
        sd.wait()  # Wait until playback has finished
        print("Playback finished.")
    else:
        print("There is no audio to play.")

# Using the function to play back the recorded audio

def remove_silence_and_adjust_f0(y, f0, hop_length=512):
    # Find indices where f0 is NaN
    nan_indices = np.where(np.isnan(f0))[0]

    # Process the start and end of the audio to find silences
    first_valid_index = np.where(~np.isnan(f0))[0][0]
    last_valid_index = np.where(~np.isnan(f0))[0][-1]

    # Trim f0 array
    f0_trimmed = f0[first_valid_index:last_valid_index + 1]

    # Remove audio samples
    y_trimmed = y[first_valid_index * hop_length:last_valid_index * hop_length + 1]

    # Replace NaNs in the middle of the audio
    mid_nans = nan_indices[(nan_indices > first_valid_index) & (nan_indices < last_valid_index)]
    if len(mid_nans) > 0:
        f0[mid_nans] = np.nanmean(f0[~np.isnan(f0)])
        f0_trimmed = np.where(np.isnan(f0_trimmed), np.nanmean(f0_trimmed[~np.isnan(f0_trimmed)]), f0_trimmed)

    return y_trimmed, f0_trimmed


if __name__ == '__main__':
    sr = 48000
    target_f0 = 100
    checkpoint_path = '../logs/betaVAESynth_dynamic/version_1/checkpoints/last.ckpt'
    if not os.path.exists(checkpoint_path):
        print('No checkpoint found')
        exit()
    model_name = "config_betaVAESynth_dynamic_1"
    # model = load_model(f'../configs/{model_name}.yaml', checkpoint_path)

    audio_file = record_audio(sr)

    pyin_f0 = compute_f0(audio_file, sr)[0]

    audio_file, pyin_f0 = remove_silence_and_adjust_f0(audio_file, pyin_f0)

    play_audio(audio_file, sr)

    # pyin_f0 = np.nan_to_num(pyin_f0, nan=np.nanmean(pyin_f0))
    # audio_file = force_f0(load_audio_file(audio_path, sr), target_f0, pyin_f0, sr)
    mel_spec = compute_mel_spectrogram(audio_file, sr)
    mel_spec = normalizar_mel_spec(mel_spec)

    model = load_model(f'../configs/{model_name}.yaml', checkpoint_path)

    pred_params = []
    for index, (mel_time_instant, pyin_time_instant) in tqdm(enumerate(zip(mel_spec.T, pyin_f0))):
        # input is the current time step together with the previous one
        if index == 0:
            input = torch.cat([mel_spec[:, index].unsqueeze(1)] * 2, dim=1)
        else:
            input = mel_spec[:, index-1:index+1]

        input = input.permute(1, 0)
        input = input.unsqueeze(0)
        pred_params.append(predict_parameters(model, input)[4].detach().cpu().numpy()[0])

    denorm_params = []
    for params, pyin in zip(pred_params, pyin_f0):
        aux_denorm_params = denormalizar_params(params).tolist()
        # add at the first position the f0
        aux_denorm_params.insert(0, 1) # voiceness
        aux_denorm_params.insert(0, pyin)

        denorm_params.append(aux_denorm_params)
    size = len(denorm_params)
    denorm_params = np.array(denorm_params).T.tolist()

    # filter individually each set of params
    for i in range(len(denorm_params)):
        denorm_params[i] = savgol_filter(denorm_params[i], window_length=32, polyorder=2)
    #filtered_params = savgol_filter(denorm_params, window_length=11, polyorder=3)

    audio = generate_audio(denorm_params, size)

    play_audio(audio, sr)
    save_audio(audio, os.path.join('./pruebaaudio.wav'), sr)
    # save_params(params, 'path/to/params.npy')

