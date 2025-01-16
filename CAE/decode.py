"""
This script is independent of the rest of the infrastructure. 
It decodes data from the latent space and saves the encoded representation. 
Variables in this file do not interact with or affect variables in the rest of the codebase.
"""

from helpers import load_decoder
from autoencoder import dec_model
from prepare_data import load_encoded_data, batch_data
import numpy as np
import h5py


# load decoder
N_lat = 5
dec_path = './data/48_RE40_' + str(N_lat)  # to save model
dec_mods = load_decoder(dec_path)

# load encoded data
enc_data_path = f'./data/48_Encoded_data_Re40_{N_lat}.h5'
U_enc = load_encoded_data(enc_data_path)

batch_size = 10
n_batch = len(U_enc) // batch_size
U_enc = batch_data(U_enc, batch_size, n_batch)

# forward pass through decoder in batches
U_dec = np.zeros((n_batch, batch_size, 48, 48, 2))
for i, batch in enumerate(U_enc):
    U_dec[i] = dec_model(batch, dec_mods)

# save decoded data
dec_file = f'./data/48_Decoded_data_Re40_{N_lat}.h5'
hf = h5py.File(dec_file,'w')
hf.create_dataset('U_dec',data=U_dec)
hf.close()
print(f"successfully decoded data saved in {dec_file}")