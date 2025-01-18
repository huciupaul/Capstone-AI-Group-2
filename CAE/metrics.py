import tensorflow as tf


# Compute the NRMSE for the test data by comparing the encoded data and the ground truth data
def compute_nrmse(U_test, U_pred):
    """
    Compute the Normalized Root-Mean-Square Error (NRMSE) for batched flowfields.

    Args:
        U_test: Tensor of shape [n_batches, batch_size, 48, 48, 2], the ground truth flowfield.
        U_pred: Tensor of shape [n_batches, batch_size, 48, 48, 2], the predicted (reconstructed) flowfield.

    Returns:
        nrmse: A scalar Tensor, the computed NRMSE across all batches.
    """

    # Ensure both tensors have the same dtype
    U_test = tf.cast(U_test, dtype=tf.float32)
    U_pred = tf.cast(U_pred, dtype=tf.float32)

    # Compute the mean squared error (MSE) for each batch
    axes = [2, 3, 4]  # reduces the 48, 48, 2 dimensions

    mse = tf.reduce_mean(tf.square(U_pred - U_test), axis=axes)  # Shape: [n_batches, batch_size]

    # Compute the variance of the ground truth (σ²) for normalization
    variance = tf.reduce_mean(tf.square(U_test - tf.reduce_mean(U_test, axis=axes, keepdims=True)), axis=axes)

    # Compute the NRMSE for each batch
    nrmse_per_time = tf.sqrt(mse / variance)  # Shape: [n_batches, batch_size]
    # print(f"Max NRMSE: {tf.reduce_max(nrmse_per_time).numpy()}")

    # Average NRMSE across the batch
    nrmse = tf.reduce_mean(nrmse_per_time)

    return nrmse