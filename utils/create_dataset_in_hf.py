import json
from tqdm import tqdm
from datasets import load_dataset

param_names = [
    'frequency', 'voiceness', 'tongue_index', 'tongue_diam',
    'lip_diam', 'constriction_index', 'constriction_diam', 'throat_diam']


def create_metadata_from_json_to_csv(output_dir, json_file):
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    with open(output_dir, 'w') as f:
        f.write("file_name,frequency,voiceness,tongue_index,tongue_diam,lip_diam,constriction_index,constriction_diam,throat_diam\n")
        test = True
        for i, (audio_name, parameters) in tqdm(enumerate(metadata.items())):
            if i < 80_000:
                prefix = "train/"
            else:
                prefix = "test/"
            parametros = ",".join([str(p) for p in parameters])
            f.write(f"{prefix}{audio_name},{parametros}\n")


if __name__ == '__main__':
    # create_metadata_from_json_to_csv("../../pt_dataset/metadata.csv", "../../pt_dataset/params.json")

    ds = load_dataset("audiofolder", data_dir="../../pt_dataset")

    print('hei')
