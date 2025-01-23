# Precursor identification of extreme events in turbulence
This project aims to identify precursor clusters to extreme events using the Kolmogorov flow as a proof of concept. The codebase consists of three main components. Namely, data generation of snapshots of a velocity vector space, a multiscale convolutional autoencoder to reduce the dimensionality of the data, and a modularity based clustering algorithm to identify precursor clusters. The codebase is written in Python and TensorFlow. This project is part of TU Delft course TI3165TU - Capstone Applied AI project, and is supervised by Nguyen Anh Khoa Doan.

### Requirements
This code is tested on TensorFlow version 2.10 and python 3.9. Requirements can be installed using:
```bash
pip install -r requirements.txt
```

## Data generation
The data generation is done using the KolSol python library, which is a Kolmogorov flow solver. The Kolmogorov flow is a two-dimensional flow that . The solver provides numerical solutions to the divergence-free Navier-Stokes equations:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial u}{\partial t} %2B u \cdot \nabla u = -\nabla p %2B \frac{1}{Re} \nabla^2 u">

## Autoencoder
The convolutional autoencoder (CAE) codebase consists of multiple python scripts. `cae_main.py` is the main file that creates the encoder and decoder modules, trains the CAE and performs validation. It calls functions from `autoencoder.py`, `constant.py`, `helpers.py`, `illustrate_atuoencoder`, `metrics.py`, `prepare_data.py`, `train.py` and `visualization.py`. Constants can be altered in `constants.py` to alter the specifics of the model architecture. Batch sizes and latent space dimension can be changed in `cae_main.py`.

Model tuning is performed in `hyperparameter_tuning.py` and the output is saved to `hyperparameter_tuning.txt`. Then, for encoding and decoding the data `encode.py` and `decode.py` are for general usage. For our specific use case, we have prepared the data for modularity based clustering in `encode_clustering.py` and `decode_clustering.py`.

### Architecture
Explaining the autoencoder architecture.

### Training
In the training loop, the autoencoder is trained on the snapshots of the velocity vector space. The autoencoder is trained to minimize the mean squared error between the input and the output. The training loop is implemented in the `placeholder` function in `placeholder.py`. Explain how the logic with learning rate adjustment and early stopping works.

### Testing

### Decoding


## Clustering

## Aknowledgments
We would like to express our gratitude to those who contributed to the success of this project.

We acknowledge the [foundational work for the autoencoder](https://github.com/MagriLab/CAE-ESN-Kolmogorov/blob/main/tutorial/CAE-ESN.ipynb) by Alberto Racca, upon which the autoencoder code is adapted. We also drew inspiration from the accompanied research paper ["Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network"](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/predicting-turbulent-dynamics-with-the-convolutional-autoencoder-echo-state-network/1E0F75CD94FCB3A1354A09622F8D25CD) authored by Alberto Racca, Nguyen Anh Khoa Doan, and Luca Magri. For the modularity based clustering, ... was used as a reference.

Special thanks to our supervisor, Nguyen Anh Khoa Doan, for providing guidance and invaluable insights throughout the project. We would also like to thank, Alexandra ..., for her support and assistance. Their contributions and expertise were instrumental in the completion of this work.