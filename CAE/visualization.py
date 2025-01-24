import matplotlib.pyplot as plt
import numpy as np
import h5py
from typing import Tuple, Optional


def plot_training_curve(vloss_plot: np.ndarray, tloss_plot: np.ndarray, epoch: int, n_lat) -> None:
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
    plt.savefig(f'MSE_{epoch}_{n_lat}.pdf')
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
    plt.savefig('hyperparameter_tuning.pdf')

    # Show the plot
    plt.show()
