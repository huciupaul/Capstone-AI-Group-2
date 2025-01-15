import os
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py

def plot_training_curve(vloss_plot, tloss_plot, N_check, epoch):
    print("Plotting...")
    plt.rcParams["figure.figsize"] = (15,4)
    plt.rcParams["font.size"]  = 20
    plt.title('MSE convergence')
    plt.yscale('log')
    plt.grid(True, axis="both", which='both', ls="-", alpha=0.3)
    plt.plot(tloss_plot[np.nonzero(tloss_plot)], 'y', label='Train loss')
    plt.plot(np.arange(np.nonzero(vloss_plot)[0].shape[0])*N_check,
                vloss_plot[np.nonzero(vloss_plot)], label='Val loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'MSE{epoch}.pdf')
    plt.show()
    print(vloss_plot)
    print(tloss_plot)
    print("End of plotting")

def save_mse_plot(vloss_plot, tloss_plot, save_path):
    """
        Saves validation loss and training loss arrays into an HDF5 file.

        Parameters:
            vloss_plot (np.ndarray): Array containing validation loss values.
            tloss_plot (np.ndarray): Array containing training loss values.
            file_path (str): The path to save the HDF5 file (default is "mse_plot.h5").
        """
    # Open an HDF5 file in write mode
    with h5py.File(save_path, "w") as hf:
        # Save the validation loss array
        hf.create_dataset("validation_loss", data=vloss_plot)
        # Save the training loss array
        hf.create_dataset("training_loss", data=tloss_plot)
    print(f"Loss plots saved successfully to {save_path}")