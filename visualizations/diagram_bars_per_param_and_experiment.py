import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error

from visualizations import visualization_utils

# Define the root directory where the data lives
root_dir = "../../generated_samples"

# Definitions for the experiment and network configurations
experiments = {
    "dynamic_10changes": "fast change",
    "dynamic": "smooth change",
    "": "static"  # directory with no extra suffix for static
}

networks = ["betaVAESynth", "encodec", "wav2vec"]

params_names = [
    'tongue_index', 'tongue_diam',
    'lip_diam', 'constriction_index', 'constriction_diam', 'throat_diam'
]

# List to store the data
data = []

# Iterate over each network and experiment type to load the data
for network in networks:
    for suffix, experiment_type in experiments.items():
        # Build the path to the specific directory
        dir_path = f"{root_dir}/{network}_{suffix}_version_1" if suffix else f"{root_dir}/{network}_version_1"

        if not os.path.exists(dir_path):
            dir_path = f"{root_dir}/{network}_{suffix}_version_0" if suffix else f"{root_dir}/{network}_version_0"

        # Check whether the directory exists
        if os.path.exists(dir_path):
            # Lists to store the data of each parameter separately
            all_preds = [[] for _ in range(6)]
            all_trues = [[] for _ in range(6)]

            # Iterate over the sample subdirectories
            for sample_dir in os.listdir(dir_path):
                if sample_dir.startswith("sample_"):
                    sample_path = os.path.join(dir_path, sample_dir)
                    param_pred_path = os.path.join(sample_path, "paramspred.npy")
                    param_true_path = os.path.join(sample_path, "paramstrue.npy")

                    # Load the arrays
                    if os.path.exists(param_pred_path) and os.path.exists(param_true_path):
                        y_pred = np.load(param_pred_path)
                        y_true = np.load(param_true_path)

                        y_pred = visualization_utils.normalizar_params(y_pred)
                        y_true = visualization_utils.normalizar_params(y_true)

                        # Distribute the data of each parameter
                        for i in range(6):  # Assumes there are 6 parameters
                            all_preds[i].append(y_pred[i])
                            all_trues[i].append(y_true[i])

            # Convert the lists to single arrays and compute the MSE for each parameter
            for i in range(6):
                if all_preds[i] and all_trues[i]:
                    correction = 1
                    if not suffix:
                        correction = 0.5
                    errors = [abs(a - b) * correction for a, b in zip(all_preds[i], all_trues[i])]
                    for error in errors:
                        data.append({
                            "Experiment": experiment_type,
                            "Network": network,
                            "Parameter": params_names[i],
                            "Error": error
                        })

# Build a DataFrame for the results
import pandas as pd

df = pd.DataFrame(data, columns=['Experiment', 'Network', 'Parameter', 'Error'])

good_names = {"betaVAESynth": "VAE+Projector", "encodec": "EnCodec", "wav2vec": "Wav2Vec"}
df["Network"] = df["Network"].apply(lambda x: good_names[x])

# Visualization setup with Seaborn
sns.set(style="whitegrid")

# Create a figure to hold the subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Set the overall figure size
fig.suptitle('Error Metrics for Each Parameter Across Experiments and Networks', fontsize=16)

sns.set_context("talk")
plt.figure(figsize=(18, 12))
for i, param in enumerate(params_names, start=1):
    plt.subplot(2, 3, i)
    plt.ylim(0, 0.3)
    sns.violinplot(x="Experiment", y="Error", hue="Network", data=df[df["Parameter"] == param], split=True, inner="quart")
    plt.title(f"Distribution of Errors for {f'{param}'}")
    # plt.xlabel("Experiment Type")
    plt.ylabel("Absolute Error")
    plt.legend(title="Network")

plt.tight_layout()
plt.show()

# tilt the x-axis labels a bit
plt.figure(figsize=(18, 10))
for i, param in enumerate(params_names, start=1):
    plt.subplot(2, 3, i)
    # add a fixed range to the y axis between 0 and 1
    plt.ylim(0, 0.3)
    param_name = param.replace('_', ' ').replace('diam', 'diameter')
    sns.boxplot(x="Experiment", y="Error", hue="Network", data=df[df["Parameter"] == param])
    plt.title(f"Box Plot of Errors for {f'{param_name}'}")
    plt.xlabel("")
    plt.xticks(rotation=-30)
    plt.ylabel("Absolute Error")
    plt.legend(title="Network")

plt.tight_layout()
plt.show()