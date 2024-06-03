import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

vae_params = pd.read_csv('../../tensorboard/params_betaVAESynth_dynamic_version_1.csv')
synth_params = pd.read_csv('../../tensorboard/params_synth_dynamic_version_0.csv')
vae_recon = pd.read_csv('../../tensorboard/recon_betaVAESynth_dynamic_version_1.csv')
synth_recon = pd.read_csv('../../tensorboard/recon_betaVAE_version_1.csv')

vae_params['model'] = 'VAE+Projector'
synth_params['model'] = 'Projector'
vae_recon['model'] = 'VAE+Projector'
synth_recon['model'] = 'VAE'

params = pd.concat([vae_params, synth_params])
recon = pd.concat([vae_recon, synth_recon])

color_dict = {
    'VAE+Projector': '#5975A4',
    'Projector': 'red',
    'VAE': 'purple'
}

sns.set_theme(style="whitegrid")
sns.set_context("talk")
plt.figure(figsize=(8, 4))
plt.ylim(0, 4)
plt.xlim(0, 20000)

# lineplot
sns.lineplot(data=params, x='Step', y='Value', hue='model', palette=color_dict)
plt.title('Spectrogram reconstruction loss over time')
# titulo x
plt.ylabel('MSE')
plt.show()

plt.figure(figsize=(8, 4))
plt.ylim(0, 5)
plt.xlim(0, 20000)
# lineplot
sns.lineplot(data=recon, x='Step', y='Value', hue='model', palette=color_dict)
plt.title('Parameter reconstruction loss over time')
plt.ylabel('Huber loss')
plt.xlabel('Step')
plt.show()
