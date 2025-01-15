import os
import h5py
from pathlib import Path
import numpy as np


def load_data(path, data_len=12000, downsample=5, transient=200):
    hf = h5py.File(path, 'r')
    print("Contents of the HDF5 file:", list(hf.keys()))

    U = np.array(hf.get('velocity_field')[transient:transient + data_len:downsample], dtype=np.float32)
    hf.close()
    print('Data Loaded successfully! \n')
    print(f'total samples: {U.shape[0]} \n')
    print(f'Data Shape: {U.shape}')

    return U

def split_batch_data(U, batch_size=40, n_batches=(40, 10, 10)):
    """
    Splits the dataset U into training, validation, and test batches.

    Parameters:
        U (np.ndarray): The input dataset of shape (total_samples, height, width, channels).
        batch_size (int): The size of each batch.
        n_batches (tuple): Number of batches for (training, validation, testing).

    Returns:
        tuple: (U_train, U_val, U_test) datasets split into batches.
    """

    print(f"Number of batches [train, val, test]: {n_batches}")
    required_samples = sum(n_batches) * batch_size

    if U.shape[0] < required_samples:
        raise ValueError(f"Not enough samples in U ({U.shape[0]}) for the requested batches ({required_samples}).")

    offset_val = n_batches[0] * batch_size
    offset_test = offset_val + n_batches[1] * batch_size

    U_train = np.zeros((n_batches[0], batch_size, U.shape[1], U.shape[2], U.shape[3]))
    U_val = np.zeros((n_batches[1], batch_size, U.shape[1], U.shape[2], U.shape[3]))
    U_test = np.zeros((n_batches[2], batch_size, U.shape[1], U.shape[2], U.shape[3]))

    for i in range(n_batches[0]):
        U_train[i] = U[i:n_batches[0]*batch_size:n_batches[0]].copy()

    for j in range(n_batches[1]):
        U_val[j] = U[j + offset_val:n_batches[1]*batch_size+offset_val:n_batches[1]].copy()

    for k in range(n_batches[2]):
        U_test[k] = U[k + offset_test:n_batches[2]*batch_size + offset_test:n_batches[2]].copy()


    # clear memory
    del U

    print('Data split successfully! \n')
    print(f"Data shape [train, val, test]: [{U_train.shape}, {U_val.shape}, {U_test.shape}]")
    return U_train, U_val, U_test


def batch_enc_data(U_all, batch_size):
    pass

def batch_dec_data(U_enc, batch_size):
    pass




