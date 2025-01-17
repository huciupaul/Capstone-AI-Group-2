"""
This script is independent of the rest of the infrastructure.
It encodes data into a latent space and saves the encoded representation.
Variables in this file do not interact with or affect variables in the rest of the codebase.
"""

from helpers import load_encoder
from autoencoder import enc_model
from prepare_data import load_velocity_clustering, load_dissip_clustering, batch_data
import numpy as np
import h5py
from constants import *

# load encoder
enc_path = './data/48_RE40_' + str(n_lat)  # to save model
enc_mods = load_encoder(enc_path)


# Path to the dataset
data_path = r"change to new path"

# Load data
U = load_velocity_clustering(data_path)
batch_size = 200
n_batch = 60

# batch U
U = batch_data(U, batch_size, n_batch)

# forward pass through encoder in batches, save without batches
U_enc = np.zeros((n_batch * batch_size, n_lat))
for i in range(n_batch):
    U_enc[i * batch_size: (i + 1) * batch_size] = enc_model(U[i], enc_mods)

# Create time array
t = np.arange(0, len(U_enc) * 0.5, 0.5)

# load dissipation
D = load_dissip_clustering(data_path)

print('U_enc shape:', U_enc.shape)
print('t shape: ', t.shape)
print('D shape: ', D.shape)

# Save encoded data with time field
enc_file = f'./data/48_Encoded_data_Re40_{n_lat}_17_1.h5'
with h5py.File(enc_file, 'w') as hf:
    hf.create_dataset('U_enc', data=U_enc)
    hf.create_dataset('dissipation_rate', data=D)
    hf.create_dataset('t', data=t)