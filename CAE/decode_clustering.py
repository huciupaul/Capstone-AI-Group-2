"""
This script is independent of the rest of the infrastructure.
It decodes the data output from the clustering algorithm.
Variables in this file do not interact with or affect variables in the rest of the codebase.
"""

from helpers import load_encoder
from autoencoder import dec_model
from prepare_data import batch_data, load_encoded_data_clustering
import numpy as np
import h5py
from constants import *


def decodeclustering(n_lat: int) -> None:
    """
    Decodes encoded velocity data using a pre-trained decoder model.

    Args:
        n_lat (int): The number of latent space dimensions.

    Output:
        Saves the decoded velocity data (`U_dec`) and corresponding time (`t`) to an HDF5 file.
    """
    # Load encoder
    dec_path = f'./Data/48_RE40_{n_lat}'  # Path to the saved decoder model
    dec_mods = load_encoder(dec_path, n_lat)

    # Path to the dataset
    data_path = r"path to encoded data"

    # Load velocity data
    U_enc, t = load_encoded_data_clustering(data_path)  # U_enc: (num_samples, latent_dim), t: (num_samples,)
    batch_size: int = 200
    n_batch: int = len(U_enc) // batch_size

    U_enc = batch_data(U_enc, batch_size, n_batch)  # U_enc shape after batching: (n_batch, batch_size, latent_dim)

    # Initialize unbatched output array (decoded velocity field)
    U_dec: np.ndarray = np.zeros((n_batch * batch_size, 48, 48, 2))  # Shape: (total_samples, height, width, channels)
    start: int = 0  # Start index for unbatching

    for batch in U_enc:
        # Decode the batch, `dec_model` returns an array of shape (batch_size, 48, 48, 2)
        decoded_batch: np.ndarray = dec_model(batch, dec_mods)

        # Compute the range of indices to fill in U_dec
        end: int = start + decoded_batch.shape[0]
        U_dec[start:end] = decoded_batch  # Assign decoded batch to the corresponding slice
        start = end  # Update the starting index for the next batch

    print('U_enc shape:', U_enc.shape)  # Expected: (n_batch, batch_size, latent_dim)
    print('t shape:', t.shape)  # Expected: (total_samples,)

    # Save decoded data with time field
    enc_file: str = f'./data/48_Decoded_data_Re40_{n_lat}_23_1.h5'
    with h5py.File(enc_file, 'w') as hf:
        hf.create_dataset('U_enc', data=U_enc)  # Saves the batched encoded data
        hf.create_dataset('t', data=t)  # Saves the corresponding time values

# n_lat: int = 10
# decodeclustering(n_lat)
