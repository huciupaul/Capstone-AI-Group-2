from autoencoder import create_enc_mods, create_dec_mods, cae_model
from train import training_loop
from prepare_data import *
from metrics import compute_nrmse 

# os.environ["OMP_NUM_THREADS"] = '15' # set cores for numpy
# os.environ['TF_INTER_OP_PARALLELISM_THREADS'] = '15' # set cores for TF
# os.environ['TF_INTRA_OP_PARALLELISM_THREADS'] = '15'


data_path = r'mylocation/Generated_data.h5'

# get data
U = load_data(data_path)

train_batches = 400
val_batches = 99
test_batches = 100

n_batches = (train_batches, val_batches, test_batches)
U_train, U_val, U_test = split_batch_data(U, n_batches=n_batches)


train_batches, batch_size, N_x, N_y, n_comp = U_train.shape
p_crop = N_x

n_lat = 5

# create encoder modules
enc_mods, ker_size, N_layers = create_enc_mods()

# explicitly obtain the size of the latent space
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

# create decoder modules
dec_mods = create_dec_mods(conv_out_size, conv_out_shape)

# train the model
n_epochs = 500
enc_mods, dec_mods = training_loop(U_train, U_val, n_epochs, enc_mods, dec_mods)

# evaluate the model
test_batches = len(U_test)
batch_size = U_test.shape[1]
U_pred = np.zeros((test_batches, batch_size, N_x, N_y, n_comp))
for i in range(test_batches):
    U_pred[i] = cae_model(U_test[i], enc_mods, dec_mods, is_train=False)[-1]

nrmse = compute_nrmse(U_test, U_pred)
print("Average NRMSE:", nrmse.numpy())
