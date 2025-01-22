"""
For illustrative purposes, this script plots five samples from the first batch of the test set, showing:
the true velocity field, the autoencoded velocity field, and the error between the two. 
"""

import numpy as np
import matplotlib.pyplot as plt
import textwrap

from constants import N_x, N_y
from autoencoder import cae_model
from helpers import load_decoder, load_encoder


# Function to wrap text for pretrier display of text in subplots
def wrap_text(text, width=25):
    return "\n".join(textwrap.wrap(text, width))


def illustrate_autoencoder(N_lat, U_test):
    path = './data/48_RE40_'+str(N_lat)
    enc_mods_test, dec_mods_test = load_encoder(path, N_lat), load_decoder(path, N_lat)
    # U_test = get_u_test_for_illustration()

    # grid
    X = np.linspace(0, 2 * np.pi, N_x) 
    Y = np.linspace(0, 2 * np.pi, N_y) 
    XX = np.meshgrid(X, Y, indexing='ij')

    # plot n snapshots and their reconstruction in the test set.
    n_snapshots = 5
    plt.rcParams["figure.figsize"] = (15, 4 * n_snapshots)
    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(n_snapshots, 3)

    # start after validation set

    for i in range(n_snapshots):
        print(f"Plotting snapshot {i+1}/{n_snapshots}")
        # testing data
        skips = len(U_test) // n_snapshots
        u = U_test[i*skips:1+i*skips].copy()
        vmax = u.max()
        vmin = u.min()
        
        # truth
        ax_truth = plt.subplot(n_snapshots, 3, i * 3 + 1)
        CS0 = ax_truth.contourf(XX[0], XX[1], u[0, :, :, 0],
                                levels=10, cmap='coolwarm', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(CS0, ax=ax_truth)
        CS = ax_truth.contour(XX[0], XX[1], u[0, :, :, 0],
                            levels=10, colors='black', linewidths=.5, linestyles='solid',
                            vmin=vmin, vmax=vmax)
        title = wrap_text(f'True velocity field at snapshot {i+1}')
        ax_truth.set_title(title, pad=20, fontsize=12)

        # autoencoded
        ax_auto = plt.subplot(n_snapshots, 3, i * 3 + 2)
        u_dec = cae_model(u, enc_mods_test, dec_mods_test)[1][0].numpy()
        CS = ax_auto.contourf(XX[0], XX[1], u_dec[:, :, 0],
                            levels=10, cmap='coolwarm', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(CS, ax=ax_auto)
        CS = ax_auto.contour(XX[0], XX[1], u_dec[:, :, 0],
                            levels=10, colors='black', linewidths=.5, linestyles='solid',
                            vmin=vmin, vmax=vmax)
        title = wrap_text(f'Autoencoded velocity field at snapshot {i+1}')
        ax_auto.set_title(title, pad=20, fontsize=12)
        
        # error
        ax_err = plt.subplot(n_snapshots, 3, i * 3 + 3)
        u_err = np.abs(u_dec - u[0]) / (vmax - vmin)
        nmae = u_err[:, :, 0].mean()

        CS = ax_err.contourf(XX[0], XX[1], u_err[:, :, 0], levels=10, cmap='coolwarm')
        cbar = plt.colorbar(CS, ax=ax_err)
        CS = ax_err.contour(XX[0], XX[1], u_err[:, :, 0], levels=10, colors='black', linewidths=.5,
                            linestyles='solid')
        title = wrap_text(f'Error between velocity fields at snapshot {i+1} with NMAE: {nmae:.4f}')
        ax_err.set_title(title, pad=20, fontsize=12)
        
    # Adjust spacing between plots
    fig.tight_layout(pad=1.0)  # Increase the padding between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Add extra spacing between rows and columns

    path = './data/48_RE40_'+str(N_lat)
    plt.savefig(path + f'/vel_fields_gt_vs_ae_n_lat{N_lat}.pdf')
    plt.show()