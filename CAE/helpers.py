from pathlib import Path
import h5py
import tensorflow as tf 
from autoencoder import PerPad2D
from constants import ker_size, n_parallel
from typing import List, Tuple


def save_cae(model_path: str, enc_mods: List[tf.keras.Model], dec_mods: List[tf.keras.Model], n_lat: int) -> None:
    """
    Saves the encoder and decoder models together with their weights to the specified path.
    Creates the directory if it does not exist.

    Args:
        model_path (str): Path where models and weights should be saved.
        enc_mods (List[tf.keras.Model]): List of encoder models.
        dec_mods (List[tf.keras.Model]): List of decoder models.
        n_lat (int): Number of latent space dimensions.

    Output:
        Saves encoder and decoder models as .h5 files, along with their weights.
    """
    print('Saving Model...')
    Path(model_path).mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    for i in range(n_parallel):
        enc_mods[i].compile()
        dec_mods[i].compile()

        enc_mods[i].save(f"{model_path}/enc_mod{ker_size[i]}_{n_lat}.h5")
        dec_mods[i].save(f"{model_path}/dec_mod{ker_size[i]}_{n_lat}.h5")

        enc_mods[i].save_weights(f"{model_path}/enc_mod{ker_size[i]}_{n_lat}_weights.h5")
        dec_mods[i].save_weights(f"{model_path}/dec_mod{ker_size[i]}_{n_lat}_weights.h5")

    print("Model saved.")


def save_optimizer_params(path: str, optimizer: tf.keras.optimizers.Optimizer) -> None:
    """
    Saves the optimizer's weights and learning rate to a file in the specified path.

    Args:
        path (str): Path where optimizer parameters should be saved.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer instance whose parameters need to be saved.

    Output:
        Saves optimizer weights and learning rate as an HDF5 file.
    """
    min_weights = optimizer.get_weights()
    with h5py.File(f"{path}/opt_weights.h5", 'w') as hf:
        for i, weight in enumerate(min_weights):
            hf.create_dataset(f'weights_{i}', data=weight)
        hf.create_dataset('length', data=len(min_weights))
        hf.create_dataset('l_rate', data=optimizer.learning_rate.numpy())

    print("Optimizer saved.")


def load_opt_weights(
    path: str, enc_mods: List[tf.keras.Model], dec_mods: List[tf.keras.Model], n_lat: int
) -> Tuple[List[tf.keras.Model], List[tf.keras.Model]]:
    """
    Loads the saved weights for encoder and decoder models from the specified path.
    Returns the encoder and decoder models with the updated weights.

    Args:
        path (str): Path where the saved weights are stored.
        enc_mods (List[tf.keras.Model]): List of encoder models.
        dec_mods (List[tf.keras.Model]): List of decoder models.
        n_lat (int): Number of latent space dimensions.

    Returns:
        Tuple[List[tf.keras.Model], List[tf.keras.Model]]: Updated encoder and decoder models with loaded weights.
    """
    print('LOADING MINIMUM')

    for i in range(n_parallel):
        enc_mods[i].load_weights(f"{path}/enc_mod{ker_size[i]}_{n_lat}_weights.h5")
        dec_mods[i].load_weights(f"{path}/dec_mod{ker_size[i]}_{n_lat}_weights.h5")

    return enc_mods, dec_mods


def load_decoder(path: str, n_lat: int) -> List[tf.keras.Model]:
    """
    Loads the saved decoder models from the specified path.
    Returns a list of compiled decoder models.

    Args:
        path (str): Path where the saved models are stored.
        n_lat (int): Number of latent space dimensions.

    Returns:
        List[tf.keras.Model]: List of compiled decoder models.
    """
    dec_mods: List[tf.keras.Model] = [None] * n_parallel

    for i in range(n_parallel):
        dec_mods[i] = tf.keras.models.load_model(
            f"{path}/dec_mod{ker_size[i]}_{n_lat}.h5",
            custom_objects={"PerPad2D": PerPad2D}
        )
        dec_mods[i].compile()

    return dec_mods


def load_encoder(path: str, n_lat: int) -> List[tf.keras.Model]:
    """
    Loads the saved encoder models from the specified path.
    Returns a list of compiled encoder models.

    Args:
        path (str): Path where the saved models are stored.
        n_lat (int): Number of latent space dimensions.

    Returns:
        List[tf.keras.Model]: List of compiled encoder models.
    """
    enc_mods: List[tf.keras.Model] = [None] * n_parallel

    for i in range(n_parallel):
        enc_mods[i] = tf.keras.models.load_model(
            f"{path}/enc_mod{ker_size[i]}_{n_lat}.h5",
            custom_objects={"PerPad2D": PerPad2D}
        )
        enc_mods[i].compile()

    return enc_mods
