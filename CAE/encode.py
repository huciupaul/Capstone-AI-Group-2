from helpers import load_encoder
from autoencoder import enc_model
from prepare_data import load_data, batch_data
import numpy as np
import h5py


# load encoder
N_lat = 5
enc_path = './data/48_RE40_' + str(N_lat)  # to save model
enc_mods = load_encoder(enc_path)

# load data
data_path = 'something.h5'
U = load_data(data_path)

# batch data
batch_size = 10
n_batch = len(U) // batch_size
U = batch_data(U, batch_size, n_batch)

# forward pass through encoder in batches, save without batches
U_enc = np.zeros((n_batch*batch_size, 48, 48, 2))
for i in range(n_batch):
    U_enc[i * batch_size: (i + 1) * batch_size + 1] = enc_model(U[i], enc_mods)

# save encoded data
enc_file = f'./data/48_Encoded_data_Re40_{N_lat}.h5'
hf = h5py.File(enc_file,'w')
hf.create_dataset('U_enc',data=U_enc)
hf.close()
print(f"successfully encoded data saved in {enc_file}")