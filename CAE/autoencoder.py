import tensorflow as tf
from constants import *
from typing import List, Tuple


def periodic_padding(image: tf.Tensor, padding: int = 1, asym: bool = False) -> tf.Tensor:
    """
    Applies periodic padding (same as np.pad('wrap')) to mimic periodic boundary conditions.

    Args:
        image (tf.Tensor): The input tensor (batch_size, height, width, channels).
        padding (int): Number of padding pixels.
        asym (bool): If True, adds an extra column/row on the right and lower edges.

    Returns:
        tf.Tensor: Padded image tensor.
    """
    if asym:
        lower_pad = image[:, :padding + 1, :]
    else:
        lower_pad = image[:, :padding, :]

    if padding != 0:
        upper_pad = image[:, -padding:, :]
        partial_image = tf.concat([upper_pad, image, lower_pad], axis=1)
    else:
        partial_image = tf.concat([image, lower_pad], axis=1)

    if asym:
        right_pad = partial_image[:, :, :padding + 1]
    else:
        right_pad = partial_image[:, :, :padding]

    if padding != 0:
        left_pad = partial_image[:, :, -padding:]
        padded_image = tf.concat([left_pad, partial_image, right_pad], axis=2)
    else:
        padded_image = tf.concat([partial_image, right_pad], axis=2)

    return padded_image


class PerPad2D(tf.keras.layers.Layer):
    """
    Periodic Padding layer.
    """
    def __init__(self, padding: int = 1, asym: bool = False, **kwargs):
        self.padding = padding
        self.asym = asym
        super(PerPad2D, self).__init__(**kwargs)

    def get_config(self) -> dict:
        """Returns the configuration of the layer to allow saving/loading."""
        config = super(PerPad2D, self).get_config()
        config.update({'padding': self.padding, 'asym': self.asym})
        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Applies periodic padding to the input tensor."""
        return periodic_padding(x, self.padding, self.asym)


def create_enc_mods(n_lat: int) -> List[tf.keras.Model]:
    """
    Creates encoder models with different kernel sizes.

    Args:
        n_lat (int): Number of latent space dimensions.

    Returns:
        List[tf.keras.Model]: List of encoder models.
    """
    enc_mods = [None] * n_parallel

    for i in range(n_parallel):
        enc_mods[i] = tf.keras.Sequential(name='Enc_' + str(i))

    # Generate encoder layers
    for j in range(n_parallel):
        for i in range(n_layers):
            # Stride=2 padding and convolution
            enc_mods[j].add(PerPad2D(padding=p_size[j], asym=True,
                                     name=f'Enc_{j}_PerPad_{i}'))
            enc_mods[j].add(tf.keras.layers.Conv2D(filters=n_fil[i], kernel_size=ker_size[j],
                                                   activation=act, padding=pad_enc, strides=2,
                                                   name=f'Enc_{j}_ConvLayer_{i}'))

            # Stride=1 padding and convolution
            if i < n_layers - 1:
                enc_mods[j].add(PerPad2D(padding=p_fin[j], asym=False,
                                         name=f'Enc_{j}_Add_PerPad1_{i}'))
                enc_mods[j].add(tf.keras.layers.Conv2D(filters=n_fil[i],
                                                       kernel_size=ker_size[j],
                                                       activation=act, padding=pad_dec, strides=1,
                                                       name=f'Enc_{j}_Add_Layer1_{i}'))

        # Fully connected layers
        enc_mods[j].add(tf.keras.layers.Flatten(name=f'Enc_{j}_Flatten'))
        enc_mods[j].add(tf.keras.layers.Dense(n_hidden, activation=act, name=f'Enc_{j}_Dense1'))
        enc_mods[j].add(tf.keras.layers.Dense(n_lat, activation=act, name=f'Enc_{j}_Dense2'))

    return enc_mods


def create_dec_mods(conv_out_size: int, conv_out_shape: Tuple[int, int, int]) -> List[tf.keras.Model]:
    """
    Creates decoder models.

    Args:
        conv_out_size (int): Number of output neurons in the first fully connected layer.
        conv_out_shape (Tuple[int, int, int]): Shape of the tensor after reshaping.

    Returns:
        List[tf.keras.Model]: List of decoder models.
    """
    dec_mods = [None] * n_parallel

    for i in range(n_parallel):
        dec_mods[i] = tf.keras.Sequential(name=f'Dec_{i}')

    # Generate decoder layers
    for j in range(n_parallel):
        dec_mods[j].add(tf.keras.layers.Dense(n_hidden, activation=act, name=f'Dec_{j}_Dense1'))
        dec_mods[j].add(tf.keras.layers.Dense(conv_out_size, activation=act, name=f'Dec_{j}_Dense2'))
        dec_mods[j].add(tf.keras.layers.Reshape(conv_out_shape, name=f'Dec_{j}_Reshape'))

        for i in range(n_layers):
            # Initial padding of latent space
            if i == 0:
                dec_mods[j].add(PerPad2D(padding=p_dec, asym=False, name=f'Dec_{j}_PerPad_{i}'))

            # Transpose convolution with stride = 2
            dec_mods[j].add(tf.keras.layers.Conv2DTranspose(filters=n_dec[i],
                                                            kernel_size=ker_size[j],
                                                            activation=act, padding=pad_dec, strides=2,
                                                            name=f'Dec_{j}_ConvLayer_{i}'))

            # Convolution with stride=1
            if i < n_layers - 1:
                dec_mods[j].add(tf.keras.layers.Conv2D(filters=n_dec[i],
                                                       kernel_size=ker_size[j],
                                                       activation=act, padding=pad_dec, strides=1,
                                                       name=f'Dec_{j}_ConvLayer1_{i}'))

        # Final layers
        dec_mods[j].add(tf.keras.layers.CenterCrop(p_crop + 2 * p_fin[j],
                                                   p_crop + 2 * p_fin[j],
                                                   name=f'Dec_{j}_Crop_{i}'))
        dec_mods[j].add(tf.keras.layers.Conv2D(filters=n_comp,
                                               kernel_size=ker_size[j],
                                               activation='linear', padding=pad_dec, strides=1,
                                               name=f'Dec_{j}_Final_Layer'))

    return dec_mods


def cae_model(inputs, enc_mods, dec_mods, is_train=False):
    """
    Forward pass of Multiscale Autoencoder.

    Args:
        inputs (tf.Tensor): Input tensor.
        enc_mods (List[tf.keras.Model]): List of encoder models.
        dec_mods (List[tf.keras.Model]): List of decoder models.
        is_train (bool): Whether the model is in training mode.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Encoded and decoded representations.
    """
    # sum of the contributions of the different CNNs
    encoded = 0
    for enc_mod in enc_mods:
        encoded += enc_mod(inputs, training=is_train)

    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(encoded, training=is_train)

    return encoded, decoded


def enc_model(U, enc_mods):
    """
    Encoder module.

    Args:
        U (tf.Tensor or np.ndarray): Input tensor.
        enc_mods (List[tf.keras.Model]): List of encoder models.

    Returns:
        tf.Tensor: Encoded representation.
    """
    encoded = 0
    for enc_mod in enc_mods:
        encoded += enc_mod(U, training=False)

    return encoded


def dec_model(encoded, dec_mods):
    """
    Decoder module.

    Args:
        encoded (tf.Tensor or nd.arrray): Encoded tensor.
        dec_mods (List[tf.keras.Model]): List of decoder models.

    Returns:
        tf.Tensor: Decoded output.
    """
    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(encoded, training=False)

    return decoded

