from autoencoder import create_enc_mods, create_dec_mods, cae_model
from train import training_loop
from prepare_data import split_batch_data, load_data
from metrics import compute_nrmse
from constants import *
import numpy as np
import tensorflow as tf
from visualization import plot_hyperparameter_tuning
import os

# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus}")
    # Enable memory growth for GPUs to avoid OOM errors
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. TensorFlow will run on the CPU.")

# Path to the dataset
fld = './Data/' #Path to folder with generated data
data_path = fld + 'Generated_data.h5' #Generated Data file name

# Load data
U = load_data(data_path, data_len=800, downsample=4, transient=0)

# Define training, validation, and test batches
batch_size = 50
n_batches = len(U) // batch_size
train_batches = int(n_batches*0.75)
val_batches = int(n_batches*0.25)

batches = (train_batches, val_batches)
U_train, U_val = split_batch_data(U, batch_size=batch_size, batches=batches)


def tune(U_train, U_val, n_lat):
    # create encoder modules
    enc_mods = create_enc_mods(n_lat)

    # explicitly obtain the size of the latent space
    # using U_val instead of train to save on computation
    output = U_val[0]
    for i, layer in enumerate(enc_mods[-1].layers):
        output = layer(output)  # Forward pass through the current layer
        if i == (n_layers - 1) * 4 + 1:  # Stop after the 4th layer (index 3)
            conv_out_shape = output.shape[1:]
            conv_out_size = np.prod(conv_out_shape)
            print("Output shape of the last convolutional layer:", conv_out_shape)
            # print("Size of last convolutional output: ", conv_out_size)
        elif i == (n_layers - 1) * 4 + 2 + 2:
            print("Size of the latent space:", output.shape[-1])

    # create decoder modules
    dec_mods = create_dec_mods(conv_out_size, conv_out_shape)

    # train the model
    n_epochs = 70
    enc_mods, dec_mods = training_loop(U_train, U_val, n_epochs, enc_mods, dec_mods, n_lat)

    U_pred_train = np.zeros((train_batches, batch_size, N_x, N_y, n_comp))
    for i in range(train_batches):
        U_pred_train[i] = cae_model(U_train[i], enc_mods, dec_mods, is_train=False)[-1]

    nrmse_train = compute_nrmse(U_train, U_pred_train)
    print("Average NRMSE train:", nrmse_train.numpy())

    U_pred_val = np.zeros((val_batches, batch_size, N_x, N_y, n_comp))
    for i in range(val_batches):
        U_pred_val[i] = cae_model(U_val[i], enc_mods, dec_mods, is_train=False)[-1]

    nrmse_val = compute_nrmse(U_val, U_pred_val)
    print("Average NRMSE val:", nrmse_val.numpy())

    return nrmse_train, nrmse_val


n_lats = [4, 8]
path_folder = r'./Plots/Tuning/'
os.makedirs(path_folder, exist_ok=True)
hfile = path_folder + "hyperparameter_tuning.txt"

with open(hfile, "w") as f:
    f.write("n_lat,nrmse_train,nrmse_val\n")  # Header
    for n_lat in n_lats:
        print("TRAINING latent size:", n_lat)
        nrmse_train, nrmse_val = tune(U_train, U_val, n_lat)

        f.write(f"{n_lat},{nrmse_train},{nrmse_val}\n")  # Write as CSV row


plot_hyperparameter_tuning(hfile)
