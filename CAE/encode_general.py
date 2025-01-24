"""
This script is independent of the rest of the infrastructure. 
It encodes data into a latent space and saves the encoded representation. 
Variables in this file do not interact with or affect variables in the rest of the codebase.
"""

from helpers import load_encoder
from autoencoder import enc_model
from prepare_data import load_data, batch_data
import numpy as np
import h5py
from constants import *


def encode(n_lat):
    """
    Encodes velocity data into a latent space using a pre-trained encoder model and
    saves the encoded representation

    Args:
        n_lat (int): Number of latent space dimensions.

    Output:
        Saves the encoded velocity data (`U_enc`) to an HDF5 file.
    """
    # load encoder
    enc_path = f'./Data/48_RE40_' + str(n_lat)  # to save model
    enc_mods = load_encoder(enc_path, n_lat)

    # load data
    data_path = f"./Data/Generated_data.h5" 
    U = load_data(data_path)

    # batch data
    batch_size = 200
    n_batch = len(U) // batch_size
    U = batch_data(U, batch_size, n_batch)

    # forward pass through encoder in batches, save without batches
    U_enc = np.zeros((n_batch * batch_size, n_lat))
    for i in range(n_batch):
        U_enc[i * batch_size: (i + 1) * batch_size] = enc_model(U[i], enc_mods)

    print(U_enc.shape)

    # save encoded data
    enc_file = f'./Data/48_Encoded_data_Re40_{n_lat}.h5'
    hf = h5py.File(enc_file, 'w')
    hf.create_dataset('U_enc', data=U_enc)
    hf.close()
    print(f"Successfully encoded data saved in {enc_file}")