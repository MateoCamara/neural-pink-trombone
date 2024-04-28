import os
import numpy as np
import matplotlib.pyplot as plt

def load_npy(file):
    data = np.load(file)
    print(f"Loaded {file} with shape {data.shape} and data type {data.dtype}")
    return data

def plot_comparison(true_data, pred_data, title):
    plt.figure(figsize=(10, 4))
    plt.plot(true_data, label='True')
    plt.plot(pred_data, label='Pred', linestyle='--')
    plt.title(f'Comparison of True and Pred Data - {title}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_mu_logvar(mu_data, logvar_data, title):
    plt.figure(figsize=(10, 4))
    plt.plot(mu_data, label='Mu')
    plt.plot(np.exp(logvar_data), label='Sigma', linestyle='--')
    plt.title(f'Mu and Sigma from LogVar - {title}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def main():
    # Load data
    path = os.path.join('..', 'generated_samples')

    paramstrue = load_npy(os.path.join(path, 'sample_0_paramstrue.npy'))
    paramspred = load_npy(os.path.join(path, 'sample_0_paramspred.npy'))
    mu = load_npy(os.path.join(path, 'sample_0_mu.npy'))
    logvar = load_npy(os.path.join(path, 'sample_0_logvar.npy'))

    # Plot comparisons
    plot_comparison(paramstrue, paramspred, 'Parameters')
    plot_mu_logvar(mu, logvar, 'Mu and Sigma')

if __name__ == '__main__':
    main()
