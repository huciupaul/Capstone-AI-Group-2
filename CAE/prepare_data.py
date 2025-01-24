import h5py
import numpy as np
from typing import Tuple, Union


def load_data(path: str, data_len: int = 120000, downsample: int = 4, transient: int = 0) -> np.ndarray:
    """
    Loads velocity field data from an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.
        data_len (int): Number of data samples to load.
        downsample (int): Downsampling factor.
        transient (int): Number of transient samples to skip.

    Returns:
        np.ndarray: Loaded velocity field data with shape (samples, height, width, channels).
    """
    with h5py.File(path, 'r') as hf:
        U = np.array(hf.get('velocity_field')[transient:transient + data_len:downsample], dtype=np.float32)

    print('Data Loaded successfully! \n')
    print(f'Total samples: {U.shape[0]} \n')
    print(f'Data Shape: {U.shape}')

    return U


def load_velocity_clustering(path: str, data_len: int = 15000) -> np.ndarray:
    """
    Loads the most recent velocity field data from an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.
        data_len (int): Number of data samples to load.

    Returns:
        np.ndarray: Velocity field data with shape (samples, height, width, channels).
    """
    with h5py.File(path, 'r') as hf:
        U = np.array(hf.get('velocity_field')[-data_len:], dtype=np.float32)

    print('Velocity Data Loaded successfully! \n')
    print(f'Total samples: {U.shape[0]} \n')
    print(f'Data Shape: {U.shape}')

    return U


def load_dissip_clustering(path: str, data_len: int = 15000) -> np.ndarray:
    """
    Loads dissipation rate data from an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.
        data_len (int): Number of data samples to load.

    Returns:
        np.ndarray: Dissipation rate data with shape (samples,).
    """
    with h5py.File(path, 'r') as hf:
        D = np.array(hf.get('dissipation_rate')[-data_len:], dtype=np.float32)

    print('Dissipation Rate Loaded successfully! \n')
    print(f'Total samples: {D.shape[0]} \n')
    print(f'Data Shape: {D.shape}')

    return D


def load_encoded_data(path: str, field: str = 'U_enc') -> np.ndarray:
    """
    Loads encoded data from an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.
        field (str): Name of the field to extract

    Returns:
        np.ndarray: Encoded data array.

    Raises:
        KeyError: If 'U_enc' dataset is not found in the file.
    """
    with h5py.File(path, 'r') as hf:
        if field in hf:
            U_enc = hf[field][:]
            print(f"Successfully read U_enc from {path}")
            return U_enc
        else:
            raise KeyError(f"Dataset '{field}' not found in the file: {path}")


def split_batch_data(
    U: np.ndarray, batch_size: int = 200, batches: Union[Tuple[int, int], Tuple[int, int, int]] = (200, 20, 20)
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Splits the dataset U into training, validation, and optionally test batches.

    Args:
        U (np.ndarray): The input dataset of shape (total_samples, height, width, channels).
        batch_size (int): The size of each batch.
        batches (tuple[int, int, int] or tuple[int, int]): Number of batches for (training, validation, testing).
                                                            or (training, validation)

    Returns:
        tuple: (U_train, U_val) or (U_train, U_val, U_test), depending on the number of batch splits.

    Raises:
        ValueError: If there are not enough samples to create the requested batches.
    """
    print(f"Number of batches [train, val]: {batches}")
    required_samples = sum(batches) * batch_size

    if U.shape[0] < required_samples:
        raise ValueError(f"Not enough samples in U ({U.shape[0]}) for the requested batches ({required_samples}).")

    offset_val = batches[0] * batch_size

    U_train = np.zeros((batches[0], batch_size, *U.shape[1:]))
    U_val = np.zeros((batches[1], batch_size, *U.shape[1:]))

    print('Batching train...')
    for i in range(batches[0]):
        U_train[i] = U[i * batch_size:(i + 1) * batch_size].copy()

    print('Batching val...')
    for j in range(batches[1]):
        U_val[j] = U[offset_val + j * batch_size:offset_val + (j + 1) * batch_size].copy()

    if len(batches) == 3:
        offset_test = offset_val + batches[1] * batch_size
        U_test = np.zeros((batches[2], batch_size, *U.shape[1:]))
        print('Batching test...')
        for j in range(batches[2]):
            U_test[j] = U[offset_test + j * batch_size:offset_test + (j + 1) * batch_size].copy()

        del U  # Free memory
        print('Original data cleared from memory')
        print(f"Data shape [train, val, test]: [{U_train.shape}, {U_val.shape}, {U_test.shape}]")
        return U_train, U_val, U_test

    del U  # Free memory
    print('Original data cleared from memory')
    print(f"Data shape [train, val]: [{U_train.shape}, {U_val.shape}]")
    return U_train, U_val


def batch_data(U: np.ndarray, b_size: int, n_batches: int) -> np.ndarray:
    """
    Splits the dataset U into smaller batches.

    Args:
        U (np.ndarray): Input dataset of shape (samples, height, width, channels).
        b_size (int): Batch size.
        n_batches (int): Number of batches.

    Returns:
        np.ndarray: Batched data with shape (n_batches, b_size, height, width, channels).
    """
    batched_data = np.zeros((n_batches, b_size, *U.shape[1:]))
    for i in range(n_batches):
        batched_data[i] = U[i * b_size: (i + 1) * b_size]

    del U  # Free memory
    return batched_data
