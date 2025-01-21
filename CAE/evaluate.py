from helpers import load_encoder, load_decoder
from hyperparameter_tuning import n_lats
from autoencoder import cae_model
from prepare_data import load_data, batch_data
from constants import *
import numpy as np
from metrics import compute_nrmse


# Path to the dataset
data_path = r"C:\Users\Rafael Ribeiro\Desktop\Capstone\Main Folder\Capstone-AI-Group-2\CAE\data\Generated_data_96000.h5"


# Load data
U_test = load_data(data_path, data_len=5000, downsample=0, transient=120000)
batch_size = 250
n_batches = 20
U_test = batch_data(U_test, b_size=batch_size, n_batches=n_batches)


txtfile = "test_nrmse.txt"
with open(txtfile, "w") as f:
    f.write("n_lat,nrmse_test\n")  # Header
    for n_lat in n_lats:
        model_path = './data/48_RE40_' + str(n_lat)
        enc_mods = load_encoder(model_path, n_lat)
        dec_mods = load_decoder(model_path, n_lat)

        U_pred_test = np.zeros((n_batches, batch_size, N_x, N_y, n_comp))
        for i in range(n_batches):
            U_pred_test[i] = cae_model(U_test[i], enc_mods, dec_mods, is_train=False)[-1]

        nrmse_test = compute_nrmse(U_test, U_pred_test)
        print(f"Average NRMSE test for n_lat {n_lat}:", nrmse_test.numpy())

        f.write(f"{n_lat},{nrmse_test}\n")  # Write as CSV row