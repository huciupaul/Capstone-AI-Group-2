
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm  # Import for progress bar

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus}")
    # Set memory growth to avoid out-of-memory (OOM) errors
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. TensorFlow will run on the CPU.")

# Function to save reduced data using chunked loading
def save_less_data_chunked(path, data_len=600000, downsample=5, transient=200, chunk_size=10000):
    """
    Process and save a subset of the dataset with chunked loading to prevent memory issues.

    Parameters:
        path (str): Path to the original HDF5 file.
        data_len (int): Total number of samples to extract.
        downsample (int): Downsampling factor.
        transient (int): Number of initial samples to skip.
        chunk_size (int): Number of samples to process in each chunk.

    Returns:
        None
    """
    # Open the original HDF5 file
    with h5py.File(path, 'r') as hf:
        velocity_field = hf.get('velocity_field')
        total_samples = data_len // downsample
        # 600 000 // 5 = 120 000

        # Output file to save the reduced data
        output_file = f'generated_data_{data_len}_{downsample}.h5'
        with h5py.File(output_file, 'w') as hf_out:
            # Create an empty dataset to store the processed data
            dset_out = hf_out.create_dataset("velocity_field",
                                             shape=(total_samples, *velocity_field.shape[1:]),
                                             dtype=np.float32)

            # Process data in chunks with a progress bar
            for start in tqdm(range(0, total_samples, chunk_size), desc="Processing chunks", unit="chunk"):

                end = min(start + chunk_size, total_samples)
                # Extract a chunk of data and apply downsampling
                chunk = velocity_field[transient + start * downsample: transient + end * downsample: downsample]
                # Save the processed chunk to the output dataset
                dset_out[start:end] = chunk.astype(np.float32)

    print(f"Successfully saved {total_samples} samples of 'velocity_field' to {output_file}")

# Path to the dataset
data_path = r"C:\Users\Rafael Ribeiro\Desktop\Capstone\Data For Loading\Generated_data_120000.h5"

# Call the function with chunked loading
save_less_data_chunked(data_path, data_len=600000, downsample=5, transient=200, chunk_size=10000)