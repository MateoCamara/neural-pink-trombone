import os

import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io.wavfile import write

import utils.utils
from data import PTServidorDataset, Tongue


def generate_audio(params, size):
    servidor = PTServidorDataset(servidor_url='127.0.0.1', servidor_port=3000, tamano_batch=1, iteraciones=1,
                                number_of_changes=size)
    return servidor.generate_specific_audio(params, length= size / 94)

def save_audio(audio, output_path, sr):
    write(output_path, sr, audio)

if __name__ == '__main__':
    model_name = "betaVAESynth_dynamic_lengua_version_2"
    audio_path = '../../human_audios'
    save_path = '../../generated_human_audios'

    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
    for audio_file_name in tqdm(os.listdir(os.path.join(save_path, model_name))):

        if not audio_file_name.endswith('.npy'):
            continue

        if audio_file_name != "ieaou_filtered.npy":
            continue


        param_names = ['f0', 'voiceness'] + utils.utils.params_names

        denorm_params = np.load(os.path.join(save_path, model_name, audio_file_name))

        size = len(denorm_params[0])

        df = pd.DataFrame(denorm_params.T, columns=param_names)

        # filter individually each set of params
        filtered_params = []
        for i in range(len(denorm_params)):
            filtered_params.append(savgol_filter(denorm_params[i], window_length=32, polyorder=2))

        df_filtered = pd.DataFrame(np.array(filtered_params).T, columns=[i + "_filtered" for i in param_names])

        df_combined = pd.concat([df, df_filtered], axis=1)

        # Crear un gráfico por cada parámetro
        # for param in param_names:
        #     plt.figure(figsize=(10, 6))
        #     sns.lineplot(data=df_combined, x=df_combined.index, y=param, label='Normal')
        #     sns.lineplot(data=df_combined, x=df_combined.index, y=param + "_filtered", label='Filtrado')
        #     plt.title(f'Comparación de {param} original y filtrado')
        #     plt.xlabel('Índice')
        #     plt.ylabel('Valor')
        #     plt.legend()
        #     plt.show()
        # audio = generate_audio(filtered_params, size)
        print('hei')
        # save_audio(audio, os.path.join(save_path, model_name, audio_file_name[:-4] + '_filtered.wav'), 48000)

        tongue = Tongue()

        diameters = np.linspace(tongue.min_diam, tongue.max_diam, 100)
        index_ranges = []

        for d in diameters:
            tongue.set_diameter(d)
            index_min, index_max = tongue.get_index_range_based_on_diam()
            index_ranges.append((index_min, index_max))

        index_mins, index_maxs = zip(*index_ranges)  # Descomprime las tuplas en dos listas separadas

        # plt.figure(figsize=(10, 6))
        # plt.plot(diameters, index_mins, '-o', label='Minimum Index')
        # plt.plot(diameters, index_maxs, '-o', label='Maximum Index')
        # plt.fill_between(diameters, index_mins, index_maxs, color='gray', alpha=0.5, label='Index Range')
        # plt.scatter(filtered_params[3], filtered_params[2], c=np.linspace(0,1, len(filtered_params[3])),
        #                                                                   cmap=plt.get_cmap('coolwarm'), label="generated")
        # plt.title('Tongue Position Variability in audio ' + audio_file_name[:-4])
        # plt.xlabel('Diameter')
        # plt.ylabel('Index')
        # plt.legend()
        # plt.grid(True)
        #
        # plt.show()
        # save
        # plt.savefig(os.path.join(save_path, model_name, audio_file_name[:-4] + '_tongue.png'))

        import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        # Configuración inicial de la figura y el eje 3D
        # sns.set_theme(style="whitegrid")
        sns.set_context("notebook")
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_zlim([1, 1.7])  # Ajusta según tus datos



        # Suponiendo que diameters, index_mins, y index_maxs están definidos
        # Plot de las líneas para mínimo y máximo índice
        ax.plot(diameters, index_mins, zs=np.min(filtered_params[4]), color='purple', zdir='z', label='Minimum Index', marker='o', markersize=1)
        ax.plot(diameters, index_maxs, zs=np.min(filtered_params[4]), color='purple', zdir='z', label='Maximum Index', marker='o', markersize=1)

        ax.plot(diameters, index_mins, zs=np.max(filtered_params[4]), color='purple', zdir='z', label='Minimum Index', marker='o', markersize=1)
        ax.plot(diameters, index_maxs, zs=np.max(filtered_params[4]), color='purple', zdir='z', label='Maximum Index', marker='o', markersize=1)

        # Relleno entre las líneas en 3D (esto es más complicado y requiere manipulación manual)
        # Debido a la complejidad, puedes optar por omitir esta parte o necesitarás crear un malla y usar plot_surface para un efecto similar

        # Datos para el scatter en 3D
        x = np.array(filtered_params[3])
        y = np.array(filtered_params[2])
        z = np.array(filtered_params[4])
        colors = np.linspace(0, 8.7, len(x))

        # Scatter en 3D con gradiente de color
        sc = ax.scatter(x, y, z, c=colors, cmap='coolwarm', label='Generated')


        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_y = y[sorted_indices]
        sorted_z = z[sorted_indices]

        # Dibuja una línea continua a lo largo de los puntos ordenados proyectada hasta z=0
        ax.scatter(np.ones_like(sorted_x)*2, sorted_y, sorted_z, color='gray', alpha=0.2, marker='o', s=2)

        for (i, j, k) in zip(x, y, z):
            ax.plot([2, i], [j, j], [k, k], 'gray', linewidth=0.1, linestyle='--', alpha=0.2)

        ax.scatter(sorted_x, np.ones_like(sorted_y)*30, sorted_z, color='gray', alpha=0.2, marker='o', s=2)

        for (i, j, k) in zip(x, y, z):
            ax.plot([i, i], [30, j], [k, k], 'gray', linewidth=0.1, linestyle='--', alpha=0.2)

        ax.scatter(sorted_x, sorted_y, np.ones_like(sorted_z)*min(sorted_z), color='gray', alpha=0.2, marker='o', s=2)

        for (i, j, k) in zip(x, y, z):
            ax.plot([i, i], [j, j], [min(sorted_z), k], 'gray', linewidth=0.1, linestyle='--', alpha=0.2)



        # Crear una barra de color
        # poner la barra abajo en horizontal
        # tamaño pequeño
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.1, orientation='horizontal')

        cbar.set_label('Time (in seconds)')


        # Configuración de etiquetas y título
        ax.set_xlabel('Diameter')
        ax.set_ylabel('Index')
        ax.set_zlabel('Lips')
        ax.set_title('Tongue and lips Position /ieaou/ sound')

        # Mostrar leyenda y cuadrícula
        # ax.legend()
        ax.grid(True)

        # Mostrar el gráfico
        plt.show()

