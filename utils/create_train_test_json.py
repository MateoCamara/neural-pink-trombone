import json
import os

path = '../../neural-pink-trombone-data/pt_dataset'

# Load the original JSON file
with open(os.path.join(path, 'params.json'), 'r') as file:
    data = json.load(file)

sorted_keys = sorted(data.keys())

# Split the keys into training and test sets
train_keys = sorted_keys[:8000]  # First 8000 keys for training
test_keys = sorted_keys[8000:]   # Remaining keys for testing

# Build the training and test dictionaries
train_data = {key: data[key] for key in train_keys}
test_data = {key: data[key] for key in test_keys}

# Save the training data to train.json
with open(os.path.join(path, 'train.json'), 'w') as train_file:
    json.dump(train_data, train_file, indent=4)  # Indent for a more readable format

# Save the test data to test.json
with open(os.path.join(path, 'test.json'), 'w') as test_file:
    json.dump(test_data, test_file, indent=4)
