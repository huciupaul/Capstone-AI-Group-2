last_conv_dep = 2  # output depth of last conv layer. To include D or vorticity, increase
n_fil = [6, 12, 24, last_conv_dep]  # number of filters encoder
n_parallel = 3  # number of parallel CNNs for multiscale
ker_size = ((3, 3), (5, 5), (7, 7))  # kernel sizes
n_layers = 4  # number of layers in every CNN
act = 'tanh'  # activation function

pad_enc = 'valid'  # no padding in the conv layer
pad_dec = 'valid'
p_size = [0, 1, 2]  # stride = 2 periodic padding size
p_fin = [1, 2, 3]  # stride = 1 periodic padding size
n_dec = [24, 12, 6, 3]  # number of filters decoder
p_dec = 1  # padding in the first decoder layer

# n_lat = 8
n_hidden = 12
p_crop = 48
n_comp = 2
N_x = 48
N_y = 48