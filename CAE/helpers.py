from pathlib import Path
import h5py
import tensorflow as tf 
from autoencoder import PerPad2D


def save_cae(model_path, enc_mods, dec_mods, ker_size, n_lat):
    n_parallel = len(enc_mods)
    print('Saving Model..')
    Path(model_path).mkdir(parents=True, exist_ok=True)  # creates directory even when it exists
    for i in range(n_parallel):
        enc_mods[i].save(model_path + '/enc_mod'+str(ker_size[i])+'_'+str(n_lat)+'.h5')
        dec_mods[i].save(model_path + '/dec_mod'+str(ker_size[i])+'_'+str(n_lat)+'.h5')
        enc_mods[i].save_weights(model_path + '/enc_mod'+str(ker_size[i])+'_'+str(n_lat)+'_weights.h5')
        dec_mods[i].save_weights(model_path + '/dec_mod'+str(ker_size[i])+'_'+str(n_lat)+'_weights.h5')
    print("Model saved.")


def save_optimizer_params(path, optimizer):
    min_weights = optimizer.get_weights()
    hf = h5py.File(path + '/opt_weights.h5','w')
    for i in range(len(min_weights)):
        hf.create_dataset('weights_'+str(i),data=min_weights[i])
    hf.create_dataset('length', data=i)
    hf.create_dataset('l_rate', data=optimizer.learning_rate)
    hf.close()
    print("Optimizer saved.")


def load_opt_weights(model_path, n_parallel, ker_size, n_lat, enc_mods, dec_mods):
    print('LOADING MINIMUM')

    for i in range(n_parallel):
        enc_mods[i].load_weights(model_path + '/enc_mod' + str(ker_size[i]) + '_' + str(n_lat) + '_weights.h5')
        dec_mods[i].load_weights(model_path + '/dec_mod' + str(ker_size[i]) + '_' + str(n_lat) + '_weights.h5')

    return enc_mods, dec_mods


def load_decoder(path, ker_size=((3, 3), (5, 5), (7, 7)), n_parallel=3, n_lat=5):
    dec_mods = [None] * n_parallel

    for i in range(n_parallel):
        dec_mods[i] = tf.keras.models.load_model(path + '/dec_mod' + str(ker_size[i]) + '_' + str(n_lat) + '.h5',
                                                 custom_objects={"PerPad2D": PerPad2D})
    return dec_mods


def load_encoder(path, ker_size=((3, 3), (5, 5), (7, 7)), n_parallel=3, n_lat=5):
    enc_mods = [None] * n_parallel

    for i in range(n_parallel):
        enc_mods[i] = tf.keras.models.load_model(path + '/enc_mod' + str(ker_size[i]) + '_' + str(n_lat) + '.h5',
                                                 custom_objects={"PerPad2D": PerPad2D})
    return enc_mods
