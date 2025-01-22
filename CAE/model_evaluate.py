from helpers import load_encoder, load_decoder
from autoencoder import cae_model
from prepare_data import load_data, batch_data
from constants import *
import numpy as np
from metrics import compute_nrmse


# Path to the dataset
data_path = r"C:\Users\edlyn\Downloads\Generated_data.h5"
n_lats = [4, 8, 12, 32, 64]


# Load data
U_test = load_data(data_path, data_len=15000, downsample=1, transient=120000)
batch_size = 250
n_batches = 20
U_test = batch_data(U_test, b_size=batch_size, n_batches=n_batches)


txtfile = "test_nrmse.txt"
with open(txtfile, "w") as f:
    f.write("n_lat,nrmse_test\n")  # Header
    for n_lat in n_lats:
        model_path = './Data/48_RE40_' + str(n_lat)
        enc_mods = load_encoder(model_path, n_lat)
        dec_mods = load_decoder(model_path, n_lat)

        U_pred_test = np.zeros((n_batches, batch_size, N_x, N_y, n_comp))
        for i in range(n_batches):
            U_pred_test[i] = cae_model(U_test[i], enc_mods, dec_mods, is_train=False)[-1]

        nrmse_test = compute_nrmse(U_test, U_pred_test)
        print(f"Average NRMSE test for n_lat {n_lat}:", nrmse_test.numpy())

        f.write(f"{n_lat},{nrmse_test}\n")  # Write as CSV row