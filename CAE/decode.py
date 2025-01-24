"""
This script is independent of the rest of the infrastructure. 
It decodes data from the latent space and saves the encoded representation in the format for clustering.
Variables in this file do not interact with or affect variables in the rest of the codebase.
"""

from helpers import load_decoder
from autoencoder import dec_model
from prepare_data import load_encoded_data, batch_data
import numpy as np
import h5py


def decode(n_lat: int, data_path: str, field='U_enc'):
    """
    Decodes encoded velocity data using a pre-trained decoder model.

    Args:
        n_lat (int): The number of latent space dimensions.
        data_path (str): Path to the encoded data
        field (str): Name of the field where encoded data is saved

    Output:
        Saves the decoded velocity data (`U_dec`) an HDF5 file.
    """
    # load decoder
    dec_path = './Data/48_RE40_' + str(n_lat)  # to save model
    dec_mods = load_decoder(dec_path, n_lat)

    # load encoded non-batched data
    U_enc = load_encoded_data(data_path, field)

    # batch the encoded data
    batch_size = 19 #num of saved clusters
    n_batch = len(U_enc) // batch_size
    U_enc = batch_data(U_enc, batch_size, n_batch)

    # forward pass through decoder in batches
    U_dec = np.zeros((n_batch * batch_size, 48, 48, 2))  # Initialize output array
    start = 0  # Start index for output

    for batch in U_enc:
        # Decode the batch, dec_model returns an array of shape (batch_size, 48, 48, 2))
        decoded_batch = dec_model(batch, dec_mods)

        # Compute the range of indices to fill in U_dec
        end = start + decoded_batch.shape[0]
        U_dec[start:end] = decoded_batch  # Assign decoded batch to the corresponding slice
        start = end  # Update the starting index for the next batch

    print("Shape of the decoded output:", U_dec.shape)

    # save decoded data
    dec_file = f'./Data/48_Decoded_data_Re40_{n_lat}.h5'
    hf = h5py.File(dec_file,'w')
    hf.create_dataset('U_dec',data=U_dec)
    hf.close()
    print(f"successfully decoded data saved in {dec_file}")


n_lat = 10

# default data path
# enc_data_path = f'./Data/48_Encoded_data_Re40_{n_lat}.h5'

enc_data_path = r"C:\Users\Rafael Ribeiro\Documents\GitHub\Capstone-AI-Group-2\Clustering\Data\Precursor_Centroids.h5"
decode(n_lat, enc_data_path, field='Precursor_Centroids')