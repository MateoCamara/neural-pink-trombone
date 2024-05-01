import os
import json

from tqdm import tqdm

from interpolate_parameters import interpolate_params

for split in ['train', 'test']:
    json_file = os.path.join("../../neural-pink-trombone-data/pt_dataset_dynamic_simplified", split + ".json")
    metadata = json.load(open(json_file, 'r'))
    metadata_interpolated = {k: interpolate_params(v, 48000, 1) for k, v in tqdm(metadata.items())}


    interpolated_json_file = os.path.join("../../neural-pink-trombone-data/pt_dataset_dynamic_simplified", split + "_interpolated.json")
    with open(interpolated_json_file, 'w') as f:
        json.dump(metadata_interpolated, f)
