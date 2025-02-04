
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from scipy.integrate import simps
import scipy.sparse as sp
import time
import os
import scipy.stats
import h5py
import pickle

start_time = time.time()


# =============================================================================
# Read the data
# =============================================================================

fld = './Data/' #Path to folder with encoded data
fln = fld + '48_Encoded_data_Re40_10' #Encoded Data file name

print(os.getcwd())

hf = h5py.File(fln, 'r')
x = np.array(hf.get('U_enc'))

print(f"Shape of x: {x.shape}")



def detect_extreme_events(x, ex_dim, nr_dev=7):
    """
    Detect extreme events in the dataset based on deviations in specified dimensions.

    :param x: 2D numpy array of data points (rows are points, columns are dimensions).
    :param ex_dim: List of dimensions used to define extreme events.
    :param nr_dev: Threshold in multiples of standard deviation for extreme event detection.
    :return: extreme_flags (1D boolean array where True indicates an extreme point)
    """
    # Initialize an array to store flags for extreme events
    extreme_flags = np.zeros(x.shape[0], dtype=bool)

    for dim in ex_dim:
        # Calculate the mean and standard deviation for the current dimension
        mean_val = np.mean(x[:, dim])
        std_dev = np.std(x[:, dim])

        # Flag data points that exceed the threshold
        extreme_flags |= abs(x[:, dim]) >= mean_val + nr_dev * std_dev

    return extreme_flags

# =============================================================================
# Normalize the data
# =============================================================================
scaler = StandardScaler()
x_normalized = scaler.fit_transform(x)

# Dimensions used to define extreme events
ex_dim = [x_normalized.shape[1] - 1] 
nr_dev = 3.5  # Set the threshold for extreme events

# K-Means Clustering
n_clusters = 125
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=400)
labels = kmeans.fit_predict(x)

# Detect extreme events
extreme_flags = detect_extreme_events(x, ex_dim, nr_dev)

# Assign "Extreme" or "Non-Extreme" label based on extreme events
cluster_names = {idx: "Non-Extreme" for idx in range(n_clusters)}

# Determine which clusters are extreme
for cluster_idx in range(n_clusters):
    # Check if the cluster contains any extreme points
    if np.any(extreme_flags[labels == cluster_idx]):
        cluster_names[cluster_idx] = "Extreme"

# Visualize clustering with "Extreme" flagged
plt.figure(figsize=(12, 8))
colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

for cluster_idx in range(n_clusters):
    cluster_points = x_normalized[labels == cluster_idx]
    plt.scatter(
        cluster_points[:, 0],    # First column as x-axis
        cluster_points[:, -1],   # Last column as y-axis
        c=[colors[cluster_idx]], label=cluster_names[cluster_idx] if cluster_idx < 20 else None,
        alpha=0.6, edgecolor="black", s=20
    )

# Highlight extreme events explicitly
plt.scatter(
    x_normalized[extreme_flags, 0],
    x_normalized[extreme_flags, -1],
    c='yellow', label="Extreme Events", edgecolor="black", s=50, marker='*'
)

plt.title("K-Means Clustering with Extreme Events Flagged (125 Clusters)")
plt.xlabel("x_0")
plt.ylabel("x_extreme")
plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=True)
plt.grid(True)
plt.show()
