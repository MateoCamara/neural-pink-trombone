import os
import librosa
import soundfile as sf


def process_audio_files(source_dir, target_dir, target_sr=48000):
    # Create the target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over all subdirectories in the source directory
    for subdir in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, subdir)):
            subdir_path = os.path.join(source_dir, subdir)
            target_subdir_path = os.path.join(target_dir, subdir)

            # Create subdirectories in the target directory if they do not exist
            if not os.path.exists(target_subdir_path):
                os.makedirs(target_subdir_path)

            filenames = [f for f in os.listdir(subdir_path) if f.endswith('.wav')]

            # Determine the minimum duration of the files in the subdirectory
            for filename in filenames:
                ref_file_name = filename.replace('_filtered', '')
                human_equivalent = os.path.join(human_directory_visqol, ref_file_name)
                if not os.path.exists(human_equivalent):
                    continue
                human_audio, sr = librosa.load(human_equivalent, sr=48000, mono=True)
                duration_human = len(human_audio)

                file_path = os.path.join(subdir_path, filename)
                audio, sr = librosa.load(file_path, sr=48000)
                # Resample to 16 kHz
                audio = librosa.resample(audio, orig_sr=48000, target_sr=target_sr)
                # Make sure all files have the same duration
                audio = audio[:duration_human]
                # Save the processed file
                target_file_path = os.path.join(target_subdir_path, filename)
                sf.write(target_file_path, audio, target_sr, format='WAV', subtype='PCM_16')


source_directory = '../../generated_human_audios'
target_directory = '../../generated_human_audios_visqol'
human_directory = '../../human_audios'
human_directory_visqol = '../../human_audios_visqol'

for human_audio in os.listdir(human_directory):
    audio, sr = librosa.load(os.path.join(human_directory, human_audio), sr=48000)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
    sf.write(os.path.join(human_directory_visqol, human_audio), audio, 48000, format='WAV', subtype='PCM_16')

process_audio_files(source_directory, target_directory)
