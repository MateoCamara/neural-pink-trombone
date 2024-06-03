import os
import subprocess
import pandas as pd

from tqdm import tqdm


def compute_visqol(ref_file, deg_file):
    visqol_executable = "../../visqol/visqol/bazel-bin/visqol"
    model = '../../visqol/libsvm_svr_pinktrombone_model.txt'
    result = subprocess.run(
        [visqol_executable, "--reference_file", ref_file, "--degraded_file", deg_file,
         "--similarity_to_quality_model", model,
         ], capture_output=True, text=True)
    try:
        visqol_score = float(result.stdout.split('LQO:')[-1].strip())
    except ValueError:
        print(f'visqol error could not be computed: {result}')
        visqol_score = 1
    return visqol_score

if __name__ == '__main__':
    sr = 48000
    target_f0 = 100
    all_available_configs = os.listdir('../configs')
    all_available_configs = [config for config in all_available_configs if "dynamic" in config]

    audio_path = '../../human_audios_visqol'
    generated_path = '../../generated_human_audios_visqol'

    df = pd.DataFrame(columns=['model', 'audio_file', 'visqol_score'])

    for config_path in all_available_configs:
        model_name = '_'.join(config_path.split('_')[1:-1])
        version_number = "version_" + config_path.split('_')[-1].split('.')[0]
        directory_name = model_name + "_" + version_number

        for audio_file_name in tqdm(os.listdir(os.path.join(generated_path, directory_name))):

            if not audio_file_name.endswith('.wav'):
                continue

            if "filtered" not in audio_file_name:
                continue

            # remove _filtered from audio file to get the ref file only if it exists
            ref_file_name = audio_file_name.replace('_filtered', '')
            ref_file = os.path.join(audio_path, ref_file_name)

            deg_file = os.path.join(generated_path, directory_name, audio_file_name)
            visqol_score = compute_visqol(ref_file, deg_file)

            new_row = {'model': directory_name, 'audio_file': ref_file_name, 'visqol_score': visqol_score}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv('visqol_scores.csv', index=False)

