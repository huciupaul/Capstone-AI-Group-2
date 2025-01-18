import tensorflow as tf
from tqdm import trange
import numpy as np
from autoencoder import cae_model
from helpers import load_opt_weights, save_cae, save_optimizer_params
import time
from visualization import save_mse_plot
from constants import n_lat


def train_step(inputs, enc_mods, dec_mods, Loss_Mse, optimizer, train=True):

    """
    Performs one training step by minimizing the loss between the input and the autoencoded output.

    Args:
        inputs: The batched input data for training or validation.
        enc_mods: List of keras layers of the encoder.
        dec_mods: List of keras layers of the decoder.
        Loss_Mse: The Mean Squared Error loss function.
        optimizer: Optimizer used for gradient descent.
        train (bool): If True, perform gradient updates. If False, only compute the loss.

    Returns:
        loss: Computed loss for the given input batch.
    """

    # Reconstructs the decoded output from the autoencoder
    decoded = cae_model(inputs, enc_mods, dec_mods, is_train=train)[-1]

    # loss between decoded output and input
    loss = Loss_Mse(inputs, decoded)

    # compute and apply gradients inside tf.function environment for computational efficiency
    if train:
        # create a variable with all the weights to perform gradient descent on
        # appending lists is done by plus sign
        varss = [] # + Dense.trainable_weights
        for enc_mod in enc_mods:
            varss += enc_mod.trainable_weights
        for dec_mod in dec_mods:
            varss += dec_mod.trainable_weights

        # Perform gradient computation and update weights
        with tf.GradientTape() as tape:
            decoded = cae_model(inputs, enc_mods, dec_mods, is_train=train)[-1]
            loss = Loss_Mse(inputs, decoded)
        grads = tape.gradient(loss, varss)
        optimizer.apply_gradients(zip(grads, varss))

    return loss


def training_loop(U_train, U_val, n_epochs, enc_mods, dec_mods):
    
    """
    Main training loop for the convolutional autoencoder.

    Args:
        U_train: Training dataset split into batches.
        U_val: Validation dataset split into batches.
        n_epochs: Number of epochs to train.
        enc_mods: List of encoder modules.
        dec_mods: List of decoder modules.

    Returns:
        enc_mods: Updated encoder modules after training.
        dec_mods: Updated decoder modules after training.
    """
    
    # random generator for later shuffling
    rng = np.random.default_rng()

    # define loss, optimizer and initial learning rate
    Loss_Mse = tf.keras.losses.MeanSquaredError()       # Mean Squared Error as the reconstruction loss
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)  # amsgrad True for better convergence
    l_rate = 0.02  # initial learning rate
    optimizer.learning_rate = l_rate

    # quantities to check and store the training and validation loss as the training goes on
    old_loss = np.zeros(n_epochs)       # Stores training loss for learning rate adjustment
    tloss_plot = np.zeros(n_epochs)     # Training loss for plotting
    vloss_plot = np.zeros(n_epochs)     # Validation loss for plotting

   # Early stopping and learning rate adjustment hyperparameters
    N_check = 5             # Frequency (in epochs) to check convergence and validation loss
    patience = 51           # Stop training if no validation loss improvement for 'patience' epochs
    last_save = patience    # Epoch where the best model was last saved

    N_lr = 10              # Number of epochs to wait before considering learning rate reduction
    lrate_update = True     # Whether to enable learning rate adjustments
    lrate_mult = 0.75       # Factor by which to reduce the learning rate

    N_plot = 20  # Frequency (in epochs) to save loss plots
    t = time.time()  # Initialize timer for epoch timing

    # Path for saving model and optimizer weights
    model_path = './data/48_RE40_' + str(n_lat)

    # Number of batches in the training and validation datasets
    train_batches = U_train.shape[0]
    val_batches = U_val.shape[0]

    # Main training loop
    for epoch in trange(n_epochs, desc='Epochs'):

        # Early stopping check
        if epoch - last_save > patience:
            print(f'Early stopping at epoch {epoch}')
            break

        # Perform gradient descent for all the batches every epoch
        loss_0 = 0
        rng.shuffle(U_train, axis=0)  # Shuffle batches
        for j in range(train_batches):
            loss = train_step(U_train[j], enc_mods, dec_mods, Loss_Mse, optimizer)
            loss_0 += loss

        # Compute and save train loss
        # tloss_plot[epoch] = loss_0.numpy() / train_batches
        tloss_plot[epoch] = loss_0 / train_batches

        # Every 'N_check' epochs checks the convergence of the training loss and val loss
        if epoch % N_check == 0:

            # Compute Validation Loss
            loss_val = 0
            for j in range(val_batches):
                loss = train_step(U_val[j], enc_mods, dec_mods, Loss_Mse, optimizer, train=False)
                loss_val += loss

            # Save the average validation loss for this epoch
            # vloss_plot[epoch] = loss_val.numpy() / val_batches
            vloss_plot[epoch] = loss_val / val_batches

            # Decreases the learning rate if the training loss is not going down with respect to
            # N_lr epochs before
            if epoch > N_lr and lrate_update:
                # check if the training loss is smaller than the average training loss N_lr epochs ago
                tt_loss = np.mean(tloss_plot[epoch - N_lr:epoch])
                if tt_loss > old_loss[epoch - N_lr]:
                    # if it is larger, load optimal val loss weights and decrease learning rate
                    enc_mods, dec_mods = load_opt_weights(model_path, enc_mods, dec_mods)

                    optimizer.learning_rate = optimizer.learning_rate * lrate_mult
                    min_weights = optimizer.get_weights()
                    optimizer.set_weights(min_weights)
                    print('LEARNING RATE CHANGE', optimizer.learning_rate.numpy())
                    
                    # Prevent further learning rate changes for the next N_lr epochs
                    old_loss[epoch - N_lr:epoch] = 1e6

            # Store current training loss
            old_loss[epoch] = tloss_plot[epoch].copy()

            # Save best model (the one with minimum validation loss)
            if epoch > 1 and vloss_plot[epoch] < \
                    (vloss_plot[:epoch - 1][np.nonzero(vloss_plot[:epoch - 1])] if np.any(vloss_plot[:epoch - 1]) else np.array([float('inf')])).min():
                # Saving the model weights
                save_cae(model_path, enc_mods, dec_mods)

                # Saving optimizer parameters
                save_optimizer_params(model_path, optimizer)

                # store the last time the val loss has decreased for early stop
                last_save = epoch

            # Print loss values and training time (per epoch)
            print('Epoch', epoch, '; Train_Loss', f"{tloss_plot[epoch]:.4f}",
                  '; Val_Loss', f"{vloss_plot[epoch]:.4f}", '; Ratio',
                  f"{(vloss_plot[epoch]) / (tloss_plot[epoch]):.4f}")
            print(f'Time per epoch {(time.time() - t):.2f} seconds')
            print('')

            # Reset time after each epoch
            t = time.time()

        # Plot every N_plot epochs
        if (epoch % N_plot == 0) and epoch != 0:
            # plot_training_curve(vloss_plot, tloss_plot, N_check, epoch)
            save_path = f'mse_plot_{epoch}.h5'
            save_mse_plot(vloss_plot, tloss_plot, save_path)

    return enc_mods, dec_mods
