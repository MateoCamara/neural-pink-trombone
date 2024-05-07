import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error

from visualizations import visualization_utils

# Define el directorio raíz donde se encuentran los datos
root_dir = "../../generated_samples"

# Definiciones para las configuraciones de experimentos y redes
experiments = {
    "dynamic_10changes": "fast change",
    "dynamic": "smooth change",
    "": "static"  # directorio sin sufijo adicional para static
}

networks = ["betaVAESynth", "encodec", "wav2vec"]

params_names = [
    'tongue_index', 'tongue_diam',
    'lip_diam', 'constriction_index', 'constriction_diam', 'throat_diam'
]

# Lista para almacenar los datos
data = []

# Recorre cada red y cada tipo de experimento para cargar los datos
for network in networks:
    for suffix, experiment_type in experiments.items():
        # Construye el path al directorio específico
        dir_path = f"{root_dir}/{network}_{suffix}_version_0" if suffix else f"{root_dir}/{network}_version_0"

        # Verifica si el directorio existe
        if os.path.exists(dir_path):
            # Listas para almacenar los datos de cada parámetro por separado
            all_preds = [[] for _ in range(6)]
            all_trues = [[] for _ in range(6)]

            # Recorre los subdirectorios de muestras
            for sample_dir in os.listdir(dir_path):
                if sample_dir.startswith("sample_"):
                    sample_path = os.path.join(dir_path, sample_dir)
                    param_pred_path = os.path.join(sample_path, "paramspred.npy")
                    param_true_path = os.path.join(sample_path, "paramstrue.npy")



                    # Carga los arrays
                    if os.path.exists(param_pred_path) and os.path.exists(param_true_path):
                        y_pred = np.load(param_pred_path)
                        y_true = np.load(param_true_path)

                        y_pred = visualization_utils.normalizar_params(y_pred)
                        y_true = visualization_utils.normalizar_params(y_true)

                        # Distribuye los datos de cada parámetro
                        for i in range(6):  # Asume que hay 6 parámetros
                            all_preds[i].append(y_pred[i])
                            all_trues[i].append(y_true[i])

            # Convierte las listas a arrays únicos y calcula el MSE para cada parámetro
            for i in range(6):
                if all_preds[i] and all_trues[i]:
                    errors = [abs(a - b) for a, b in zip(all_preds[i], all_trues[i])]
                    for error in errors:
                        data.append({
                            "Experiment": experiment_type,
                            "Network": network,
                            "Parameter": params_names[i],
                            "Error": error
                        })

# Crea un DataFrame para los resultados
import pandas as pd

df = pd.DataFrame(data, columns=['Experiment', 'Network', 'Parameter', 'Error'])

# Configuración de la visualización con Seaborn
sns.set(style="whitegrid")

# Crea una figura para alojar los subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Configura el tamaño general de la figura
fig.suptitle('Error Metrics for Each Parameter Across Experiments and Networks', fontsize=16)
plt.figure(figsize=(18, 12))
for i, param in enumerate(params_names, start=1):
    plt.subplot(2, 3, i)
    sns.violinplot(x="Experiment", y="Error", hue="Network", data=df[df["Parameter"] == param], split=True, inner="quart")
    plt.title(f"Distribution of Errors for {f'{param}'}")
    plt.xlabel("Experiment Type")
    plt.ylabel("Absolute Error")
    plt.legend(title="Network")

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 12))
for i, param in enumerate(params_names, start=1):
    plt.subplot(2, 3, i)
    sns.boxplot(x="Experiment", y="Error", hue="Network", data=df[df["Parameter"] == param])
    plt.title(f"Box Plot of Errors for {f'{param}'}")
    plt.xlabel("Experiment Type")
    plt.ylabel("Absolute Error")
    plt.legend(title="Network")

plt.tight_layout()
plt.show()