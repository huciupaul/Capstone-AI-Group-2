import tensorflow as tf
from tqdm import trange
import numpy as np
from autoencoder import cae_model
from helpers import *
import time
from visualization import plot_training_curve

def train_step(inputs, enc_mods, dec_mods, Loss_Mse, optimizer, train=True):

    """
    Trains the model by minimizing the loss between input and output
    """

    # autoencoded field
    decoded  = cae_model(inputs, enc_mods, dec_mods, is_train=train)[-1]

    # loss with respect to the data
    loss = Loss_Mse(inputs, decoded)

    # compute and apply gradients inside tf.function environment for computational efficiency
    if train:
        # create a variable with all the weights to perform gradient descent on
        # appending lists is done by plus sign
        varss = [] # + Dense.trainable_weights
        for enc_mod in enc_mods:
            varss  += enc_mod.trainable_weights
        for dec_mod in dec_mods:
            varss +=  dec_mod.trainable_weights

        with tf.GradientTape() as tape:
            decoded  = cae_model(inputs, enc_mods, dec_mods, is_train=train)[-1]
            loss = Loss_Mse(inputs, decoded)
        grads = tape.gradient(loss, varss)
        optimizer.apply_gradients(zip(grads, varss))

    return loss


def training_loop(U_train, U_val, n_epochs, enc_mods, dec_mods, N_lat, ker_size):
    rng = np.random.default_rng()  # random generator for later shuffling

    # define loss, optimizer and initial learning rate
    Loss_Mse = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)  # amsgrad True for better convergence
    l_rate = 0.002
    optimizer.learning_rate = l_rate

    # quantities to check and store the training and validation loss and the training goes on
    old_loss = np.zeros(n_epochs)  # needed to evaluate training loss convergence to update l_rate
    tloss_plot = np.zeros(n_epochs)  # training loss
    vloss_plot = np.zeros(n_epochs)  # validation loss
    N_check = 5  # each N_check epochs we check convergence and validation loss
    patience = 20  # if the val_loss has not gone down in the last patience epochs, early stop
    last_save = patience  # last epoch where the model was saved

    # Hyperparameters for changing learning rate
    N_lr = 5
    lrate_update = True
    lrate_mult = 0.75

    N_plot = 10
    t = time.time()  # Initialize time for printing time per epoch
    model_path = './data/48_RE40_' + str(N_lat)  # to save model

    train_batches = U_train.shape[0]
    val_batches = U_val.shape[0]

    for epoch in trange(n_epochs, desc='Epochs'):

        # Incorporate early stopping
        if epoch - last_save > patience:
            print('Early stopping')
            break

        # Perform gradient descent for all the batches every epoch
        loss_0 = 0
        rng.shuffle(U_train, axis=0)  # shuffle batches
        for j in range(train_batches):
            loss = train_step(U_train[j], enc_mods, dec_mods, Loss_Mse, optimizer)
            loss_0 += loss

        # save train loss
        tloss_plot[epoch] = loss_0.numpy() / train_batches

        # every N epochs checks the convergence of the training loss and val loss
        if (epoch % N_check == 0):

            # Compute Validation Loss
            loss_val = 0
            for j in range(val_batches):
                loss = train_step(U_val[j], enc_mods, dec_mods, Loss_Mse, optimizer, train=False)
                loss_val += loss

            # Save validation loss
            vloss_plot[epoch] = loss_val.numpy() / val_batches

            # Decreases the learning rate if the training loss is not going down with respect to
            # N_lr epochs before
            if epoch > N_lr and lrate_update:
                # check if the training loss is smaller than the average training loss N_lr epochs ago
                tt_loss = np.mean(tloss_plot[epoch - N_lr:epoch])
                if tt_loss > old_loss[epoch - N_lr]:
                    # if it is larger, load optimal val loss weights and decrease learning rate
                    enc_mods, dec_mods = load_opt_weights(model_path)

                    optimizer.learning_rate = optimizer.learning_rate * lrate_mult
                    min_weights = optimizer.get_weights()  # RV - just added this line
                    optimizer.set_weights(min_weights)
                    print('LEARNING RATE CHANGE', optimizer.learning_rate.numpy())
                    old_loss[epoch - N_lr:epoch] = 1e6  # so that l_rate is not changed for N_lr steps

            # store current loss
            old_loss[epoch] = tloss_plot[epoch].copy()

            # save best model (the one with minimum validation loss)
            if epoch > 1 and vloss_plot[epoch] < \
                    (vloss_plot[:epoch - 1][np.nonzero(vloss_plot[:epoch - 1])]).min():
                # saving the model weights
                save_cae(model_path, enc_mods, dec_mods, ker_size, N_lat)

                # saving optimizer parameters
                save_optimizer_params(model_path, optimizer)

                last_save = epoch  # store the last time the val loss has decreased for early stop

            # Print loss values and training time (per epoch)
            print('Epoch', epoch, '; Train_Loss', f"{tloss_plot[epoch]:.4f}",
                  '; Val_Loss', f"{vloss_plot[epoch]:.4f}", '; Ratio',
                  f"{(vloss_plot[epoch]) / (tloss_plot[epoch]):.4f}")
            print(f'Time per epoch {(time.time() - t):.2f} seconds')
            print('')
            t = time.time()  # Reset time after each epoch

        if (epoch % N_plot == 0) and epoch != 0:  # Plot every N_plot epochs
            plot_training_curve(vloss_plot, tloss_plot, N_check, epoch)

    return enc_mods, dec_mods
