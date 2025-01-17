from autoencoder import create_enc_mods, create_dec_mods, cae_model
from train import training_loop
from prepare_data import split_batch_data, load_data
from metrics import compute_nrmse
from constants import *
import numpy as np
import tensorflow as tf

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
data_path = r"C:\Users\Rafael Ribeiro\Desktop\Capstone\New Git\Capstone-AI-Group-2\generated_data_600000_5.h5"


# Load data
U = load_data(data_path, data_len=120000, downsample=1, transient=0)

# Define training, validation, and test batches
# n_batches = 120 000 // 200 = 600
# is b_size = 100, n_batch = 1200, where 1100 for training
batch_size = 200

train_batches = 560
val_batches = 20
test_batches = 20

n_batches = (train_batches, val_batches, test_batches)
U_train, U_val, U_test = split_batch_data(U, batch_size=200, n_batches=n_batches)

# create encoder modules
enc_mods = create_enc_mods()

# explicitly obtain the size of the latent space
# using U_val instead of train to save on computation
output = U_val[0]
for i, layer in enumerate(enc_mods[-1].layers):
    output = layer(output)  # Forward pass through the current layer
    if i == (n_layers - 1) * 4 + 1:  # Stop after the 4th layer (index 3)
        conv_out_shape = output.shape[1:]
        conv_out_size = np.prod(conv_out_shape)
        print("Output shape of the last convolutional layer:", conv_out_shape)
        print("Size of last convolutional output: ", conv_out_size)
    elif i == (n_layers - 1) * 4 + 2 + 1:
        print("Size of the latent space:", output.shape[-1])

# create decoder modules
dec_mods = create_dec_mods(conv_out_size, conv_out_shape)

# train the model
n_epochs = 500
enc_mods, dec_mods = training_loop(U_train, U_val, n_epochs, enc_mods, dec_mods)

# evaluate the model
# test_batches = len(U_test)
# batch_size = U_test.shape[1]

U_pred = np.zeros((test_batches, batch_size, N_x, N_y, n_comp))
for i in range(test_batches):
    U_pred[i] = cae_model(U_test[i], enc_mods, dec_mods, is_train=False)[-1]

nrmse = compute_nrmse(U_test, U_pred)
print("Average NRMSE:", nrmse.numpy())
