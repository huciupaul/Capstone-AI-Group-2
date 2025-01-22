import h5py
import numpy as np


def load_data(path, data_len=120000, downsample=4, transient=0):
    hf = h5py.File(path, 'r')
    U = np.array(hf.get('velocity_field')[transient:transient + data_len:downsample], dtype=np.float32)
    hf.close()
    print('Data Loaded successfully! \n')
    print(f'total samples: {U.shape[0]} \n')
    print(f'Data Shape: {U.shape}')

    return U


def load_velocity_clustering(path, data_len=15000):
    hf = h5py.File(path, 'r')
    U = np.array(hf.get('velocity_field')[-data_len:], dtype=np.float32)
    hf.close()
    print('Data Loaded successfully! \n')
    print(f'total samples: {U.shape[0]} \n')
    print(f'Data Shape: {U.shape}')

    return U


def load_dissip_clustering(path, data_len=15000):
    hf = h5py.File(path, 'r')
    D = np.array(hf.get('dissipation_rate')[-data_len:], dtype=np.float32)
    hf.close()
    print('D Loaded successfully! \n')
    print(f'total samples: {D.shape[0]} \n')
    print(f'Data Shape: {D.shape}')

    return D


def load_encoded_data(path):
    with h5py.File(path, 'r') as hf:
        if 'U_enc' in hf:
            U_enc = hf['U_enc'][:]
            print(f"Successfully read U_enc from {path}")
            return U_enc
        else:
            raise KeyError(f"Dataset 'U_enc' not found in the file: {path}")


def split_batch_data(U, batch_size=200, batches=(200, 20, 20)):
    """
    Splits the dataset U into training, validation, and test batches.

    Parameters:
        U (np.ndarray): The input dataset of shape (total_samples, height, width, channels).
        batch_size (int): The size of each batch.
        batches (tuple): Number of batches for (training, validation, testing).

    Returns:
        tuple: (U_train, U_val) datasets split into batches.
    """

    print(f"Number of batches [train, val]: {batches}")
    required_samples = sum(batches) * batch_size

    if U.shape[0] < required_samples:
        raise ValueError(f"Not enough samples in U ({U.shape[0]}) for the requested batches ({required_samples}).")

    offset_val = batches[0] * batch_size

    U_train = np.zeros((batches[0], batch_size, U.shape[1], U.shape[2], U.shape[3]))
    U_val = np.zeros((batches[1], batch_size, U.shape[1], U.shape[2], U.shape[3]))

    print('batching train...')
    for i in range(batches[0]):
        U_train[i] = U[i:batches[0]*batch_size:batches[0]].copy()

    print('batching val...')
    for j in range(batches[1]):
        U_val[j] = U[j + offset_val:batches[1]*batch_size+offset_val:batches[1]].copy()

    if len(batches) == 3:
        offset_test = offset_val + batches[1] * batch_size
        U_test = np.zeros((batches[2], batch_size, U.shape[1], U.shape[2], U.shape[3]))
        print('batching test...')
        for j in range(batches[2]):
            U_test[j] = U[j + offset_test:batches[2] * batch_size + offset_test:batches[2]].copy()

    # clear memory
    del U
    print('original data cleared from memory')

    print('Data split successfully! \n')
    if len(batches) == 3:
        print(f"Data shape [train, val, test]: [{U_train.shape}, {U_val.shape}, {U_test.shape}]")
        return U_train, U_val, U_test

    else:
        return U_train, U_val


def batch_data(U, b_size, n_batches):
    batched_data = np.zeros((n_batches, b_size, *U.shape[1:]))
    for i in range(n_batches):
        batched_data[i] = U[i*b_size: (i+1)*b_size]
    del U
    return batched_data
