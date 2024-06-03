import pandas as pd


def visualize_visqol_results():
    df = pd.read_csv('../../visqol_scores.csv')
    rows_to_keep = ['betaVAESynth_dynamic_lengua_version_2',
                    "betaVAESynth_dynamic_version_1",
                    "encodec_dynamic_10changes_version_2",
                    "encodec_dynamic_version_2",
                    "wav2vec_dynamic_10changes_version_1",
                    "wav2vec_dynamic_version_2"]
    df = df[df['model'].isin(rows_to_keep)]
    df['audio_file'] = df['audio_file'].apply(lambda x: x.split('.wav')[0])

    # replace with betaVAESynth, encodec, wav2vec with smooth and fast changes
    df['Experiment type'] = df['model'].apply(lambda x: x.replace('betaVAESynth_dynamic_lengua_version_2', 'fast change')
                                   .replace('betaVAESynth_dynamic_version_1', 'smooth change')
                                   .replace('encodec_dynamic_10changes_version_2', 'fast change')
                                   .replace('encodec_dynamic_version_2', 'smooth change')
                                   .replace('wav2vec_dynamic_10changes_version_1', 'fast change')
                                   .replace('wav2vec_dynamic_version_2', 'smooth change'))

    df['model'] = df['model'].apply(lambda x: x.replace('betaVAESynth_dynamic_lengua_version_2', 'betaVAESynth')
                                      .replace('betaVAESynth_dynamic_version_1', 'betaVAESynth')
                                      .replace('encodec_dynamic_10changes_version_2', 'encodec')
                                      .replace('encodec_dynamic_version_2', 'encodec')
                                      .replace('wav2vec_dynamic_10changes_version_1', 'wav2vec')
                                      .replace('wav2vec_dynamic_version_2', 'wav2vec'))

    # remove visqol with value == 1
    df = df[df['visqol_score'] != 1]

    df['visqol_score'] = df['visqol_score'].apply(lambda x: x + 0.5)

    df = df.rename(columns={'visqol_score': 'ViSQOL score'})

    df = df.sort_values(by='model', ascending=True)

    good_names = {"betaVAESynth": "VAE+Projector", "encodec": "EnCodec", "wav2vec": "Wav2Vec"}
    df["model"] = df["model"].apply(lambda x: good_names[x])

    # draw histogram with seaborn

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    # cambia los colores a rojo y morado
    # sns.set_palette("husl")
    # cambia el tamaño de la letra
    sns.set_context("talk")
    ax = sns.histplot(data=df, x='ViSQOL score', hue='model', multiple='stack', bins=20)
    ax.set_title('Histogram of ViSQOL scores')
    plt.show()

    # draw boxplot with seaborn
    plt.figure(figsize=(8, 6))
    # sns.set_palette("husl")
    ax = sns.boxplot(data=df, x='Experiment type', y='ViSQOL score', hue='model')
    ax.set_title('Boxplot of ViSQOL scores')
    plt.show()





    # df.to_csv('visqol_scores_pivoted.csv', index=False)

    return df

if __name__ == '__main__':
    visualize_visqol_results()