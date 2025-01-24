from helpers import load_encoder, load_decoder
from autoencoder import cae_model
from prepare_data import load_data, batch_data
from constants import *  # Assumes constants like N_x, N_y, n_comp are defined
import numpy as np
from metrics import compute_nrmse
from visualization import illustrate_autoencoder

# Path to the dataset
fld = './Data/' #Path to folder with generated data
data_path = fld + 'Generated_data.h5' #Generated Data file name

n_lat_opt: int = 10  # latent space dimensions to test

# Load test dataset, 120 000 already used for training and val
U_test: np.ndarray = load_data(data_path, data_len=15000, downsample=1, transient=120000)
# Expected shape: (num_samples, N_x, N_y, n_comp)

batch_size: int = 250
n_batches: int = 20
U_test = batch_data(U_test, b_size=batch_size, n_batches=n_batches)
# After batching, shape: (n_batches, batch_size, N_x, N_y, n_comp)

# File to store NRMSE results
txtfile: str = "test_nrmse.txt"
with open(txtfile, "w") as f:
    f.write("n_lat,nrmse_test\n")  # CSV header


    model_path: str = f'./Data/48_RE40_{n_lat_opt}'  # Model directory

    # Load encoder and decoder
    enc_mods = load_encoder(model_path, n_lat_opt)
    dec_mods = load_decoder(model_path, n_lat_opt)

    # Initialize storage for predicted data
    U_pred_test: np.ndarray = np.zeros((n_batches, batch_size, N_x, N_y, n_comp))
    # Shape: (n_batches, batch_size, N_x, N_y, n_comp)

    for i in range(n_batches):
        # Forward pass through the encoder-decoder model
        U_pred_test[i] = cae_model(U_test[i], enc_mods, dec_mods, is_train=False)[-1]

    # Compute normalized root-mean-square error (NRMSE)
    nrmse_test = compute_nrmse(U_test, U_pred_test)
    print(f"Average NRMSE test for n_lat {n_lat_opt}:", nrmse_test.numpy())

    # Save results to file
    f.write(f"{n_lat},{nrmse_test}\n")  # Append NRMSE results

# plot comparison figure
illustrate_autoencoder(n_lat_opt, U_test)


