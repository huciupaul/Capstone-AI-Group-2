from autoencoder import *
# from utils.visualizaton import *
from train import *
import numpy as np
from prepare_data import *

# os.environ["OMP_NUM_THREADS"] = '15' # set cores for numpy
# os.environ['TF_INTER_OP_PARALLELISM_THREADS'] = '15' # set cores for TF
# os.environ['TF_INTRA_OP_PARALLELISM_THREADS'] = '15'



data_path = r"C:\Users\edlyn\Downloads\Generated_data.h5"

# get data
U = load_data(data_path)
N_x, N_y = U.shape[1:3]
U_train, U_val, U_test = split_batch_data(U)

Re = 40

p_crop = U_train.shape[2]
N_lat = 5
enc_mods, ker_size, N_layers = create_enc_mods(N_lat)

#explicitly obtain the size of the latent space
output = U_train[0]
for i, layer in enumerate(enc_mods[-1].layers):
    output = layer(output)  # Forward pass through the current layer
    if i == (N_layers - 1) * 4 + 1:  # Stop after the 4th layer (index 3)
        conv_out_shape = output.shape[1:]
        conv_out_size = np.prod(conv_out_shape)
        print("Output shape of the last convolutional layer:", conv_out_shape)
        print("Size of last convolutional output: ", conv_out_size)
    elif i == (N_layers - 1) * 4 + 2 + 1:
         print("Size of the latent space:", output.shape[-1])

n_comp = U_train.shape[-1]
dec_mods = create_dec_mods(conv_out_size, conv_out_shape, p_crop, n_comp)
# connected layers
# encoded, decoded = cae_model(U_train, enc_mods, dec_mods)


n_epochs = 2
enc_mods, dec_mods = training_loop(U_train, U_val, n_epochs, train_batches, enc_mods, dec_mods, N_lat, ker_size)