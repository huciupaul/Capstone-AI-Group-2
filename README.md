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

For clustering, the relevant depedencies can be found at `clustering/requirements_clustering.txt`. 
This can be installed with:

```bash 
pip install -r requirements_clustering.txt
```

## Data generation
The data generation is done using the KolSol python library, which is a Kolmogorov flow solver. The Kolmogorov flow is a two-dimensional flow that . The solver provides numerical solutions to the divergence-free Navier-Stokes equations: #TODO

## Autoencoder
The convolutional autoencoder (CAE) codebase consists of multiple python scripts. `cae_main.py` is the main file that creates the encoder and decoder modules, trains the CAE and performs validation. It calls functions from `autoencoder.py`, `constant.py`, `helpers.py`, `illustrate_autoencoder`, `metrics.py`, `prepare_data.py`, `train.py` and `visualization.py`. Constants can be altered in `constants.py` to alter the specifics of the model architecture. Batch sizes and latent space dimension can be changed in `cae_main.py`.

Model tuning is performed in `hyperparameter_tuning.py` and the output is saved to `hyperparameter_tuning.txt`. Then, for encoding and decoding the data `encode.py` and `decode.py` are for general usage. For our specific use case, we have prepared the data for modularity based clustering in `encode_clustering.py` and `decode_clustering.py`.

### Architecture
The CAE consists of an encoder and a decoder. The encoder consists of three convolutional layers with kernel sizes (3x3), (5x5), (7x7), followed by two fully connected layers to further reduce the the dimension of the data to the desired parameter `N_lat`. All the layers used `tanh` activation functions. The decoder architecture is the mirror image of the encoder. The architecture of the convolutioanl layers is directly imported from [Alberto Racca CAE], which is inspired by the work of [Hasegawa 2020](https://doi.org/10.1007/s00162-020-00528-w). The CAE architecture is defined in `autoencoder.py`.

### Training and hyperparameter tuning
In `train.py` the training loop runs all the steps required for the iterative training of the CAE model. The CAE is trained on the snapshots of the velocity vector space. At its core, it iteratively minimizes the reconstruction error (Mean Squared Error) between the input data and the autoencoder's output using the Adam optimizer.  The loop incorporates several features, including:

* Learning Rate Adaptation: The learning rate decreases dynamically by a factor `lrate_mult` when the training loss plateaus, enhancing convergence stability.
* Early Stopping: Training stops if the validation loss does not improve for a predefined `patience` number of epochs, preventing overfitting and saving computational resources.
* Validation Monitoring: The validation loss is computed every `N_check` epochs to evaluate generalization performance.
* Checkpointing: The best-performing model weights and optimizer parameters are saved based on the lowest validation loss. If the training loss does not improve compared to the average of `N_lr` previous epochs, the learning rate is adapted and the model is reset to a previous best-performing state.
* Visualization: Loss plots are saved every `N_plot` epochs to visualize training and validation trends.

For hyperparameter tuning, the `hyperparameter_tuning.py` script trains the CAE for a set of latent space, `n_lat` dimensions and saves the results to `hyperparameter_tuning.txt`. The optimal `n_lat` was selected through a trade-off between NRMSE and its size. 


### Testing
In `model_test.py` the convolutional autoencoder model's performance is evaluated. The testing dataset is loaded to `U_test`, batched and passed through the model. The predicted values are then compared to the actual test values using an NRMSE metric. This is done by the loading of encoder and decoder using the `load_encoder` and `load_decoder` functions from `helpers.py`, the NRMSE calculation is done using `compute_nrmse` from `metrics.py`.


### Encoding and decoding
The encoding and decoding of the data are done using the two functions from `encode.py` and `decode.py`. The encoder part uses the pre-trained model to lower the data dimension to the set latent space. The decoder loads the encoded data and decodes it back to its original shape.

During encoding, the model is loaded and the dataset is batched according to the preset `batch_size` and `n_batches` to be passed through the encoder. Subsequently, the encoded dataset saved in an unbatched format for general use and saved to a HDF5 file.

The decoding is the reverse process of encoding, which loads the pre-trained decoder modules, takes the encoded data, batches it, and decodes it to the original dimension and shape. Finally, the decoded dataset is created and saved to another HDF5 file for visualization.

## Clustering
Modularity-based clustering consists of six files in total. Main file is `main_with_loop_only_features.py` which uses functions defined in `clustering_func_only_features.py`, `modularity.py`, `spectralopt.py` and `_divide.py`. In the main file, after the clustering process is done, the clusters are saved to .npz files. These clusters then can be used in `main_load_clusters.py` which postprocesses, calculates average time between extreme and precursor events, detects false positives and negatives and plots phase space plot, tesselated phase space plot and Dissipation time series with background color plot. `main_load_clusters.py` also saves the precursor centroids that have been validated to lead to an extreme event as `Precursor_Centroids.h5` for the decoder.

### Implementation
The clustering implementation is distributed across multiple scripts:
- **`main_with_loop_only_features.py`**: Main script for the modularity-based clustering process, utilizing functions from supporting scripts.
- **`clustering_func_only_features.py`**: Contains helper functions for clustering, including graph transformations and calculations related to extreme events.
- **`modularity.py`**: Implements core modularity calculations and modularity matrix computations.
- **`spectralopt.py`**: Provides functions for spectral methods used to partition the network into communities.
- **`_divide.py`**: Implements the division of communities into subgroups based on modularity optimization.

The clustering algorithm takes preprocessed velocity field data as input, applies modularity maximization, and saves the resulting clusters as `.npz` files for further analysis.

---

### Features
Key features of the clustering implementation include:
1. **Spectral Clustering**: Uses spectral methods to partition the network into communities by iteratively optimizing the modularity metric.
2. **Refinement**: The clustering process includes refinement steps to enhance the modularity and ensure well-defined community structures.
3. **Analysis of Extreme Events**: Special consideration is given to precursor and extreme clusters, analyzing their relationships and transitions.
4. **Visualization**: Results are visualized as:
   - Phase space plots
   - Tessellated phase space plots
   - Dissipation time series with color-coded clusters.

---

### Post-Processing
Post-processing is handled in **`main_load_clusters.py`**, which:
- Computes the average time between precursor and extreme events.
- Identifies false positives and negatives in precursor detection.
- Generates detailed visualizations to understand the dynamics of the identified clusters.

---

### Usage
To run the clustering process:
1. Ensure the dependencies listed in `clustering/requirements_clustering.txt` are installed.
2. Use **`main_with_loop_only_features.py`** to execute the clustering pipeline. 
3. Analyze the saved clusters with **`main_load_clusters.py`** for further insights and to save the precursor cluster centroids.

### Remark
`main_with_loop_only_features.py` encounters a value error (ex: 'ValueError: 1411845100 is not in list') when running in a traditional IDE. Running in a virtual linux enviroment (we used Ubuntu terminal) resolves this error.
---


### Alternative clustering methods
To compare the modularity based clustering with other, more common clustering methods, this repository implements three alternative clustering methods, 'K-Means', 'Agglomerative clustering' and 'DBSCAN'. Which can be found in 'clustering/Alternative Clustering/Alternative clusters.py', 'clustering/Alternative Clustering/Alternative clusters agglomerative.py', 'clustering/Alternative Clustering/Alternative clusters density.py' respectively.
These clustering methods are applied to a normalized and reduced subset to reduce computing times.

### K-Means Clustering
- Partitions data into a user-defined number of clusters (k)
- Identifies extreme events by detecting points deviating significantly from the mean (±5 standard deviations)

### Agglomerative Clustering
- A hierarchical clustering method that partitions data into a user-defined number of clusters with the 'ward linkage' approach.
- Groups data into clusters based on similarity
- Identifies extreme events by detecting points deviating significantly from the mean (±5 standard deviations)

### DBSCAN Clustering
- Density-based clustering method that groups points based on local density
- Treats outliers as noise.
- Identifies extreme events by detecting points deviating significantly from the mean (±5 standard deviations)

*** Visualisation
Each file plots the clusters found, as well as the extreme events in a 2D scatter plot, showing the x_extreme against the x_0 value.


## Aknowledgments
We would like to express our gratitude to those who contributed to the success of this project.

We acknowledge the [foundational work for the autoencoder](https://github.com/MagriLab/CAE-ESN-Kolmogorov/blob/main/tutorial/CAE-ESN.ipynb) by Alberto Racca, upon which the autoencoder code is adapted. We also drew inspiration from the accompanied research paper ["Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network"](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/predicting-turbulent-dynamics-with-the-convolutional-autoencoder-echo-state-network/1E0F75CD94FCB3A1354A09622F8D25CD) authored by Alberto Racca, Nguyen Anh Khoa Doan, and Luca Magri. For the modularity based clustering, ... was used as a reference.

Special thanks to our supervisor, Nguyen Anh Khoa Doan, for providing guidance and invaluable insights throughout the project. We would also like to thank Alexandra ..., for her support and assistance. Their contributions and expertise were instrumental in the completion of this work.
