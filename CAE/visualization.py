import matplotlib.pyplot as plt
import numpy as np
import h5py
from typing import Tuple, Optional
import textwrap
from constants import N_x, N_y
from autoencoder import cae_model
from helpers import load_decoder, load_encoder


def plot_training_curve(vloss_plot: np.ndarray, tloss_plot: np.ndarray, save_path, epoch: int, n_lat) -> None:
    """
    Plots training and validation loss over epochs.

    Args:
        vloss_plot (np.ndarray): Validation loss values over epochs.
        tloss_plot (np.ndarray): Training loss values over epochs.
        epoch (int): The number of epochs, used for saving the plot.

    Output:
        Saves and displays the MSE convergence plot.
    """
    plt.rcParams["figure.figsize"] = (15, 4)
    plt.rcParams["font.size"] = 20

    # Set up the plot
    plt.title('MSE Convergence')
    plt.yscale('log')
    plt.grid(True, axis="both", which='both', alpha=0.3)

    # Training Loss
    plt.plot(tloss_plot, 'y', label='Train loss')

    # Extract nonzero validation loss values and corresponding epochs
    val_indices = np.nonzero(vloss_plot)[0]
    val_epochs = val_indices  # Convert indices to epochs
    val_loss_values = vloss_plot[val_indices]

    # Plot validation loss only at computed epochs
    plt.plot(val_epochs, val_loss_values, label='Val loss')

    last_tloss_index = np.max(np.nonzero(tloss_plot)) if np.any(tloss_plot) else len(tloss_plot) - 1
    # Set x-axis limit to match the length of tloss_plot
    plt.xlim([0, last_tloss_index])

    plt.xlabel('Epochs')
    plt.legend()

    # Save and show the plot
    plt.savefig(save_path)
    plt.show()


def save_mse_plot(vloss_plot: np.ndarray, tloss_plot: np.ndarray, save_path: str) -> None:
    """
    Saves validation loss and training loss arrays into an HDF5 file.

    Args:
        vloss_plot (np.ndarray): Array containing validation loss values.
        tloss_plot (np.ndarray): Array containing training loss values.
        save_path (str): The path to save the HDF5 file.

    Output:
        Saves loss plots in an HDF5 file at the given path.
    """
    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("validation_loss", data=vloss_plot)
        hf.create_dataset("training_loss", data=tloss_plot)

    print(f"Loss plots saved successfully to {save_path}")


def read_mse_plot(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Reads validation loss and training loss arrays from an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (vloss_plot, tloss_plot) as NumPy arrays if successful,
        otherwise returns None.
    """
    try:
        with h5py.File(file_path, 'r') as hf:
            if 'validation_loss' not in hf or 'training_loss' not in hf:
                raise KeyError("Missing dataset(s) in HDF5 file.")

            vloss_plot = np.array(hf['validation_loss'])
            tloss_plot = np.array(hf['training_loss'])

        print(f"Successfully read loss data from {file_path}")
        return vloss_plot, tloss_plot

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def plot_hyperparameter_tuning(txt_file: str) -> None:
    """
    Plots the effect of different latent sizes on the validation NRMSE.

    Args:
        txt_file (str): Path to a CSV file containing latent sizes and their corresponding NRMSE values.

    Output:
        Saves and displays the latent size vs. validation NRMSE plot.
    """
    n_lat_list = []
    nrmse_val_list = []

    with open(txt_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            n_lat, _, nrmse_val = map(float, line.strip().split(","))  # Ignore nrmse_train
            n_lat_list.append(int(n_lat))
            nrmse_val_list.append(nrmse_val)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(n_lat_list, nrmse_val_list, marker='o', linestyle='-', color='b', label='NRMSE Validation')

    plt.xlabel("Latent Size (n_lat)")
    plt.ylabel("NRMSE (Validation)")
    plt.title("Latent Size vs. Validation NRMSE")
    plt.xticks(n_lat_list)
    plt.grid(True)
    plt.legend()
    plt.savefig('./Plots/Tuning/hyperparameter_tuning.pdf')

    # Show the plot
    plt.show()


# Function to wrap text for pretrier display of text in subplots
def wrap_text(text, width=25):
    return "\n".join(textwrap.wrap(text, width))


def illustrate_autoencoder(N_lat, U_test):
    path = './Data/48_RE40_' + str(N_lat)
    enc_mods_test, dec_mods_test = load_encoder(path, N_lat), load_decoder(path, N_lat)


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
        print(f"Plotting snapshot {i + 1}/{n_snapshots}")
        # testing data
        skips = len(U_test) // n_snapshots
        u = U_test[i * skips:1 + i * skips].copy()
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
        title = wrap_text(f'True velocity field at snapshot {i + 1}')
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
        title = wrap_text(f'Decoded velocity field at snapshot {i + 1}')
        ax_auto.set_title(title, pad=20, fontsize=12)

        # error
        ax_err = plt.subplot(n_snapshots, 3, i * 3 + 3)
        u_err = np.abs(u_dec - u[0]) / (vmax - vmin)
        nmae = u_err[:, :, 0].mean()

        CS = ax_err.contourf(XX[0], XX[1], u_err[:, :, 0], levels=10, cmap='coolwarm')
        cbar = plt.colorbar(CS, ax=ax_err)
        CS = ax_err.contour(XX[0], XX[1], u_err[:, :, 0], levels=10, colors='black', linewidths=.5,
                            linestyles='solid')
        title = wrap_text(f'Error between velocity fields at snapshot {i + 1} with NMAE: {nmae:.4f}')
        ax_err.set_title(title, pad=20, fontsize=12)

    # Adjust spacing between plots
    fig.tight_layout(pad=1.0)  # Increase the padding between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Add extra spacing between rows and columns

    path = './Data/48_RE40_' + str(N_lat)
    plt.savefig(path + f'/vel_fields_gt_vs_ae_n_lat{N_lat}.pdf')
    plt.show()
