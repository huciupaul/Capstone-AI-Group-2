# Precursor identification of extreme events in turbulence
This project aims to identify precursor clusters to extreme events using the Kolmogorov flow as a proof of concept. The codebase consists of three main components. Namely, data generation of snapshots of a velocity vector space, a multiscale convolutional autoencoder (CAE) to reduce the dimensionality of the data, and a modularity based clustering algorithm to identify precursor clusters. The codebase is written in Python and TensorFlow. This project is part of TU Delft course TI3165TU - Capstone Applied AI project, and is supervised by Nguyen Anh Khoa Doan.

### Requirements
This code is tested on TensorFlow version 2.10 and python 3.9. The required packages for the data generation and the CAE can be installed with pip using:

```bash
pip install -r requirements.txt
```
Or for anaconda users with:

```bash 
conda env create -f environment.yaml
```

For clustering, the relevant depedencies can be found at `clustering/conda_pack.txt` and `clustering/tf_package_list.txt`.


## Data generation
The data generation is done using the KolSol python library, which is a Kolmogorov flow solver. The Kolmogorov flow is a two-dimensional flow that . The solver provides numerical solutions to the divergence-free Navier-Stokes equations:

## Autoencoder
The convolutional autoencoder (CAE) codebase consists of multiple python scripts. `cae_main.py` is the main file that creates the encoder and decoder modules, trains the CAE and performs validation. It calls functions from `autoencoder.py`, `constant.py`, `helpers.py`, `illustrate_atuoencoder`, `metrics.py`, `prepare_data.py`, `train.py` and `visualization.py`. Constants can be altered in `constants.py` to alter the specifics of the model architecture. Batch sizes and latent space dimension can be changed in `cae_main.py`.

Model tuning is performed in `hyperparameter_tuning.py` and the output is saved to `hyperparameter_tuning.txt`. Then, for encoding and decoding the data `encode.py` and `decode.py` are for general usage. For our specific use case, we have prepared the data for modularity based clustering in `encode_clustering.py` and `decode_clustering.py`.

### Architecture
The CAE consists of an encoder and a decoder. The encoder consists of three convolutional layers with kernel sizes (3x3), (5x5), (7x7), followed by two fully connected layers to further reduce the the dimension of the data to the desired parameter `N_lat`. The decoder architecture is the mirror image of the encoder. The architecture is inspired by the work of [Hasegawa 2020](https://doi.org/10.1007/s00162-020-00528-w). The CAE architecture is defined in `autoencoder.py`.

### Training and hyperparameter tuning
In `train.py` the training loop orchestrates the optimization of the CAE. The autoencoder is trained on the snapshots of the velocity vector space. At its core, it iteratively minimizes the reconstruction error (Mean Squared Error) between the input data and the autoencoder's output using the Adam optimizer.  The loop incorporates several features, including:

* Learning Rate Adaptation: The learning rate decreases dynamically when the training loss plateaus, enhancing convergence stability.
* Early Stopping: Training stops if the validation loss does not improve for a predefined number of epochs, preventing overfitting and saving computational resources.
* Validation Monitoring: The validation loss is computed periodically to evaluate generalization performance.
* Checkpointing: The best-performing model weights and optimizer parameters are saved based on the lowest validation loss. If the validation loss does not improve for a predefined number of epochs, the learning rate is adapted and the model is reset to a previous best-performing state.
* Visualization: Loss plots are saved at predefined intervals to visualize training and validation trends.

For hyperparameter tuning, the `hyperparameter_tuning.py` script performs a grid search over a set of latent space dimensions. The script trains the CAE for each latent space size and saves the results to `hyperparameter_tuning.txt`. The best latent space size was selected through a trade-off between reconstruction error and its size. 

### Testing

### Encoding and decoding

## Clustering
Modularity-based clustering consists of six files in total. Main file is `main_with_loop_only_features.py` which uses functions defined in `clustering_func_only_features.py`, `modularity.py`, `spectralopt.py` and `_divide.py. In the main file, after the clustering process is done, the clusters are saved to .npz files. These clusters then can be used in `main_load_clusters.py` which postprocesses, calculates average time between extreme and precursor events, detects false positives and negatives and plots phase space plot, tesselated phase space plot and Dissipation time series with background color plot. 

### Modularity-based Clustering

### Alternative clustering methods


## Aknowledgments
We would like to express our gratitude to those who contributed to the success of this project.

We acknowledge the [foundational work for the autoencoder](https://github.com/MagriLab/CAE-ESN-Kolmogorov/blob/main/tutorial/CAE-ESN.ipynb) by Alberto Racca, upon which the autoencoder code is adapted. We also drew inspiration from the accompanied research paper ["Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network"](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/predicting-turbulent-dynamics-with-the-convolutional-autoencoder-echo-state-network/1E0F75CD94FCB3A1354A09622F8D25CD) authored by Alberto Racca, Nguyen Anh Khoa Doan, and Luca Magri. For the modularity based clustering, ... was used as a reference.

Special thanks to our supervisor, Nguyen Anh Khoa Doan, for providing guidance and invaluable insights throughout the project. We would also like to thank Alexandra ..., for her support and assistance. Their contributions and expertise were instrumental in the completion of this work.