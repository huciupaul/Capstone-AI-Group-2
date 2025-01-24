"""
This script encodes data into a specific format required by the clustering algorithm.
It maps velocity data into a latent space and appends the dissipation rate before saving the encoded representation.
Variables in this file do not interact with or affect variables in the rest of the codebase.
"""

from helpers import load_encoder
from autoencoder import enc_model
from prepare_data import load_velocity_clustering, load_dissip_clustering, batch_data
import numpy as np
import h5py
from constants import dt  # Assumes constants like time step size (0.2) are defined


def encodeclustering(n_lat: int, data_path: str) -> None:
    """
    Encodes velocity data into a latent space using a pre-trained encoder model.
    Appends dissipation rate data and saves the encoded representation.

    Args:
        n_lat (int): Number of latent space dimensions.

    Output:
        Saves the encoded velocity data (`U_enc`) and corresponding time (`t`) to an HDF5 file.
    """
    # Load encoder
    enc_path = f'./Data/48_RE40_{n_lat}'  # Path to the saved encoder model
    enc_mods = load_encoder(enc_path, n_lat)

    # Load velocity data
    U: np.ndarray = load_velocity_clustering(data_path, data_len=15000)
    # Expected shape: (num_samples, N_x, N_y, n_comp)

    batch_size = 200
    n_batch = len(U) // batch_size  # Number of full batches

    # Batch U
    U = batch_data(U, batch_size, n_batch)
    # After batching: (n_batch, batch_size, N_x, N_y, n_comp)

    # Forward pass through encoder in batches, storing the encoded data
    U_enc: np.ndarray = np.zeros((n_batch * batch_size, n_lat))  # Shape: (total_samples, n_lat)

    for i in range(n_batch):
        U_enc[i * batch_size: (i + 1) * batch_size] = enc_model(U[i], enc_mods)

    # Load dissipation rate data
    D: np.ndarray = load_dissip_clustering(data_path, data_len=15000)
    # Expected shape: (num_samples,)

    # Ensure dissipation rate has the correct shape
    D = D.reshape(-1, 1)  # Reshape to (n_samples, 1) if it's a 1D array

    # Append dissipation rate as the last column of U_enc
    U_enc = np.hstack((U_enc, D))  # Final shape: (total_samples, n_lat + 1)

    # Create time array (assuming constant time step of 0.2)
    t: np.ndarray = np.arange(0, len(U_enc) * dt, dt)  # Shape: (total_samples,)

    print('U_enc shape:', U_enc.shape)  # Expected: (total_samples, n_lat + 1)
    print('t shape: ', t.shape)  # Expected: (total_samples,)

    # Save encoded data with time field
    enc_file = f'./data/48_Encoded_data_Re40_{n_lat}_23_1.h5'
    with h5py.File(enc_file, 'w') as hf:
        hf.create_dataset('U_enc', data=U_enc)  # Encoded velocity data with dissipation rate
        hf.create_dataset('t', data=t)  # Time array


# Uncomment to run the encoding function
# n_lat: int = 10
# data_path = r"C:\Users\Rafael Ribeiro\Desktop\Capstone\Main Folder\Capstone-AI-Group-2\CAE\data\Generated_data_96000.h5"
# encodeclustering(n_lat, data_path)
