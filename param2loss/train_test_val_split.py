import os
import random
import shutil

def train_test_val_split(wavs_dir, wavs_noisy_dir):
    # Get all train, test and validation paths
    train_dir = os.path.join(wavs_dir,'train')
    val_dir = os.path.join(wavs_dir,'val')
    test_dir = os.path.join(wavs_dir,'test')
    train_dir_noisy = os.path.join(wavs_noisy_dir,'train')
    val_dir_noisy = os.path.join(wavs_noisy_dir,'val')
    test_dir_noisy = os.path.join(wavs_noisy_dir,'test')

    # Create destination directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir_noisy, exist_ok=True)
    os.makedirs(val_dir_noisy, exist_ok=True)
    os.makedirs(test_dir_noisy, exist_ok=True)

    # List all files in the source directory
    files = []
    for i in os.listdir(wavs_dir):
        if not os.path.isdir(os.path.join(wavs_dir,i)):
            files.append(i)
    files_noisy = []
    for i in os.listdir(wavs_noisy_dir):
        if not os.path.isdir(os.path.join(wavs_dir,i)):
            files_noisy.append(i)

    # Shuffle the list of files
    random.shuffle(files)
    random.shuffle(files_noisy)

    # Calculate number of files for each split
    total_files = len(files)
    train_count = int(total_files * 0.8)
    val_count = int(total_files * 0.1)

    # Move files to train directory
    for file_name in files[:train_count]:
        src_path = os.path.join(wavs_dir, file_name)
        dest_path = os.path.join(train_dir, file_name)
        shutil.move(src_path, dest_path)

    # Move files to validation directory
    for file_name in files[train_count:train_count + val_count]:
        src_path = os.path.join(wavs_dir, file_name)
        dest_path = os.path.join(val_dir, file_name)
        shutil.move(src_path, dest_path)

    # Move remaining files to test directory
    for file_name in files[train_count + val_count:]:
        src_path = os.path.join(wavs_dir, file_name)
        dest_path = os.path.join(test_dir, file_name)
        shutil.move(src_path, dest_path)

    # Move noisy files to train directory
    for file_name in files[:train_count]:
        src_path = os.path.join(wavs_noisy_dir, file_name)
        dest_path = os.path.join(train_dir_noisy, file_name)
        shutil.move(src_path, dest_path)

    # Move noisy files to validation directory
    for file_name in files[train_count:train_count + val_count]:
        src_path = os.path.join(wavs_noisy_dir, file_name)
        dest_path = os.path.join(val_dir_noisy, file_name)
        shutil.move(src_path, dest_path)

    # Move remaining noisy files to test directory
    for file_name in files[train_count + val_count:]:
        src_path = os.path.join(wavs_noisy_dir, file_name)
        dest_path = os.path.join(test_dir_noisy, file_name)
        shutil.move(src_path, dest_path)


## MAIN ##

# Paths
params_json_file = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/params.json'
params_noisy_json_file = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/params_noisy.json'
wavs_dir = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/wavs'
wavs_noisy_dir = '/home/fmacias@gaps_domain.ssr.upm.es/AES_2024/neural-pink-trombone/param2loss/data_p2l/wavs_noisy'

# Split data in train, validation and test (different directories)
train_test_val_split(wavs_dir, wavs_noisy_dir)