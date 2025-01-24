"""
This script is independent of the rest of the infrastructure.
It decodes the data output from the clustering algorithm.
Variables in this file do not interact with or affect variables in the rest of the codebase.
"""

from helpers import load_encoder
from autoencoder import dec_model
from prepare_data import batch_data, load_encoded_data
import numpy as np
import h5py
import tensorflow as tf
from constants import *


def decodeclustering(n_lat: int, data_path: str, field='U_enc') -> None:
    """
    Decodes encoded velocity data using a pre-trained decoder model.

    Args:
        n_lat (int): The number of latent space dimensions.
        data_path (str): The path to the encoded data
        field (str): The field in which the data is saved

    Output:
        Saves the decoded velocity data (`U_dec`) and corresponding time (`t`) to an HDF5 file.
    """
    # Load encoder
    dec_path = f'./Data/48_RE40_{n_lat}'  # Path to the saved decoder model
    dec_mods = load_encoder(dec_path, n_lat)

    # Load velocity data
    U_enc = load_encoded_data(data_path, field=field)[:,:-1]  # discard dissipation rate
    batch_size = 19
    n_batch = len(U_enc) // batch_size

    U_enc = batch_data(U_enc, batch_size, n_batch)  # U_enc shape after batching: (n_batch, batch_size, latent_dim)

    # Initialize unbatched output array (decoded velocity field)
    U_dec = np.zeros((n_batch * batch_size, 48, 48, 2))  # Shape: (total_samples, height, width, channels)
    start = 0  # Start index for output

    for batch in U_enc:
        # Decode the batch, `dec_model` returns an array of shape (batch_size, 48, 48, 2)
        decoded_batch = dec_model(batch, dec_mods)

        # Compute the range of indices to fill in U_dec
        end = start + decoded_batch.shape[0]
        U_dec[start:end] = decoded_batch  # Assign decoded batch to the corresponding slice
        start = end  # Update the starting index for the next batch

    print('U_enc shape:', U_enc.shape)  # Expected: (n_batch, batch_size, latent_dim)

    # Save decoded data with time field
    enc_file = f'./Data/48_Decoded_data_Re40_{n_lat}.h5'
    with h5py.File(enc_file, 'w') as hf:
        hf.create_dataset('U_dec', data=U_dec)  # Saves the batched encoded data


n_lat = 10

# default data path
# enc_data_path = f'./Data/48_Encoded_data_Re40_{n_lat}.h5'

enc_data_path = f"./Data/48_Encoded_data_Re40_10.h5" #path to encoded data
decodeclustering(n_lat, enc_data_path, field='Precursor_Centroids')