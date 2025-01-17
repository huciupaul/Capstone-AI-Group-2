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

# load encoder
enc_path = './data/48_RE40_' + str(n_lat)  # to save model
enc_mods = load_encoder(enc_path)


# Path to the dataset
data_path = r"C:\Users\Rafael Ribeiro\Desktop\Capstone\CAE\generated_data_600000_5.h5"

# Load data
U = load_data(data_path, data_len=120000, downsample=1, transient=0)[-8000:]
batch_size = 200
n_batch = 40

# batch U
U = batch_data(U, batch_size, n_batch)

# forward pass through encoder in batches, save without batches
U_enc = np.zeros((n_batch * batch_size, n_lat))
for i in range(n_batch):
    U_enc[i * batch_size: (i + 1) * batch_size] = enc_model(U[i], enc_mods)

print(U_enc.shape)

# save encoded data
enc_file = f'./data/48_Encoded_data_Re40_{n_lat}_17_1.h5'
hf = h5py.File(enc_file, 'w')
hf.create_dataset('U_enc', data=U_enc)
hf.close()
print(f"Successfully encoded data saved in {enc_file}")