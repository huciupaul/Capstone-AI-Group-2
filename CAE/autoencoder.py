import tensorflow as tf


def cae_model(inputs, enc_mods, dec_mods, is_train=False):
    '''
    Multiscale autoencoder, taken from Hasegawa 2020. The contribution of the CNNs at different
    scales are simply summed.
    '''

    # sum of the contributions of the different CNNs
    encoded = 0
    for enc_mod in enc_mods:
        encoded += enc_mod(inputs, training=is_train)

    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(encoded, training=is_train)

    return encoded, decoded


def periodic_padding(image, padding=1, asym=False):
    '''
    Create a periodic padding (same of np.pad('wrap')) around the image,
    to mimic periodic boundary conditions.
    When asym=True on the right and lower edges an additional column/row is added
    '''

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
    Periodic Padding layer
    """

    def __init__(self, padding=1, asym=False, **kwargs):
        self.padding = padding
        self.asym = asym
        super(PerPad2D, self).__init__(**kwargs)

    def get_config(self):  # needed to be able to save and load the model with this layer
        config = super(PerPad2D, self).get_config()
        config.update({
            'padding': self.padding,
            'asym': self.asym,
        })
        return config

    def call(self, x):
        return periodic_padding(x, self.padding, self.asym)


def create_enc_mods(N_lat):
    last_conv_dep = 1  # output depth of last conv layer, if we want to include dissipation rate and vorticity, increase this number
    n_fil = [6, 12, 24, last_conv_dep]  # number of filters encoder
    N_parallel = 3  # number of parallel CNNs for multiscale
    ker_size = [(3, 3), (5, 5), (7, 7)]  # kernel sizes
    N_layers = 4  # number of layers in every CNN
    act = 'tanh'  # activation function

    pad_enc = 'valid'  # no padding in the conv layer
    pad_dec = 'valid'
    p_size = [0, 1, 2]  # stride = 2 periodic padding size
    p_fin = [1, 2, 3]  # stride = 1 periodic padding size

    # initialize the encoders and decoders with different kernel sizes
    enc_mods = [None] * (N_parallel)
    for i in range(N_parallel):
        enc_mods[i] = tf.keras.Sequential(name='Enc_' + str(i))

    # generate encoder layers
    for j in range(N_parallel):
        for i in range(N_layers):

            # stride=2 padding and conv
            enc_mods[j].add(PerPad2D(padding=p_size[j], asym=True,
                                     name='Enc_' + str(j) + '_PerPad_' + str(i)))
            enc_mods[j].add(tf.keras.layers.Conv2D(filters=n_fil[i], kernel_size=ker_size[j],
                                                   activation=act, padding=pad_enc, strides=2,
                                                   name='Enc_' + str(j) + '_ConvLayer_' + str(i)))

            # stride=1 padding and conv
            if i < N_layers - 1:
                enc_mods[j].add(PerPad2D(padding=p_fin[j], asym=False,
                                         name='Enc_' + str(j) + '_Add_PerPad1_' + str(i)))
                enc_mods[j].add(tf.keras.layers.Conv2D(filters=n_fil[i],
                                                       kernel_size=ker_size[j],
                                                       activation=act, padding=pad_dec, strides=1,
                                                       name='Enc_' + str(j) + '_Add_Layer1_' + str(i)))
        # Add fully connected layer
        enc_mods[j].add(tf.keras.layers.Flatten(name='Enc_' + str(j) + '_Flatten'))
        enc_mods[j].add(tf.keras.layers.Dense(N_lat, activation='linear', name='Enc_' + str(j) + '_Dense'))

    return enc_mods, ker_size, N_layers



def create_dec_mods(conv_out_size, conv_out_shape, p_crop, n_comp):
    n_dec = [24, 12, 6, 3]  # number of filters decoder
    N_parallel = 3  # number of parallel CNNs for multiscale
    ker_size = [(3, 3), (5, 5), (7, 7)]  # kernel sizes
    N_layers = 4  # number of layers in every CNN
    act = 'tanh'  # activation function
    p_dec = 1  # padding in the first decoder layer
    pad_dec = 'valid'
    p_fin = [1, 2, 3]  # stride = 1 periodic padding size

    dec_mods = [None] * (N_parallel)
    for i in range(N_parallel):
        dec_mods[i] = tf.keras.Sequential(name='Dec_' + str(i))

    # generate decoder layers
    for j in range(N_parallel):

        # Add fully connected layer first to map latent space to the appropriate dimensions
        dec_mods[j].add(tf.keras.layers.Dense(conv_out_size, activation='linear', name='Dec_' + str(j) + '_Dense'))
        dec_mods[j].add(tf.keras.layers.Reshape(conv_out_shape, name='Dec_' + str(j) + '_Reshape'))

        for i in range(N_layers):

            # initial padding of latent space
            if i == 0:
                dec_mods[j].add(PerPad2D(padding=p_dec, asym=False,
                                         name='Dec_' + str(j) + '_PerPad_' + str(i)))

                # Transpose convolution with stride = 2
            dec_mods[j].add(tf.keras.layers.Conv2DTranspose(filters=n_dec[i],
                                                            output_padding=None, kernel_size=ker_size[j],
                                                            activation=act, padding=pad_dec, strides=2,
                                                            name='Dec_' + str(j) + '_ConvLayer_' + str(i)))

            # Convolution with stride=1
            if i < N_layers - 1:
                dec_mods[j].add(tf.keras.layers.Conv2D(filters=n_dec[i],
                                                       kernel_size=ker_size[j],
                                                       activation=act, padding=pad_dec, strides=1,
                                                       name='Dec_' + str(j) + '_ConvLayer1_' + str(i)))

        # crop and final linear convolution with stride=1
        dec_mods[j].add(tf.keras.layers.CenterCrop(p_crop + 2 * p_fin[j],
                                                   p_crop + 2 * p_fin[j],
                                                   name='Dec_' + str(j) + '_Crop_' + str(i)))
        dec_mods[j].add(tf.keras.layers.Conv2D(filters=n_comp,
                                               kernel_size=ker_size[j],
                                               activation='linear', padding=pad_dec, strides=1,
                                               name='Dec_' + str(j) + '_Final_Layer'))

    return dec_mods


def cae_model(inputs, enc_mods, dec_mods, is_train=False):
    '''
    Forward pass of Multiscale autoencoder, taken from Hasegawa 2020. The contribution of the CNNs at different
    scales are simply summed.
    '''

    # sum of the contributions of the different CNNs
    encoded = 0
    for enc_mod in enc_mods:
        encoded += enc_mod(inputs, training=is_train)

    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(encoded, training=is_train)

    return encoded, decoded


def call_decoder(dec_mods):
    pass

