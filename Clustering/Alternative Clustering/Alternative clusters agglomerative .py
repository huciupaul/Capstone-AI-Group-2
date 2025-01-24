import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import h5py
import os

# =============================================================================
# Read the data
# =============================================================================

# fld = './ExampleData/'
# fln = fld + '48_Encoded_data_Re40_8_18_1.h5'
fln = r"C:\Users\agata\Downloads\48_Encoded_data_Re40_12_22_1.h5"

hf = h5py.File(fln, 'r')
x = np.array(hf.get('U_enc'))
t = np.array(hf.get('t'))

print(f"Shape of x: {x.shape}")

# Use only 20% of the data
num_elem = int(len(x) * 0.2)
x_20 = x[:num_elem]

print(f"Shape of x: {x_20.shape}")

# =============================================================================
# Normalize the data
# =============================================================================
scaler = StandardScaler()
x_normalized = scaler.fit_transform(x_20)

# Perform Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clustering.fit_predict(x_normalized)

# Cluster naming for visualization
cluster_names = {idx: "Normal or Precursor" for idx in range(3)}

# Detect extreme events based on standard deviation
def detect_extreme_events(x, ex_dim, nr_dev=5):
    extreme_flags = np.zeros(x.shape[0], dtype=bool)
    for dim in ex_dim:
        mean_val = np.mean(x[:, dim])
        std_dev = np.std(x[:, dim])
        extreme_flags |= abs(x[:, dim]) >= mean_val + nr_dev * std_dev
    return extreme_flags

ex_dim = [x_normalized.shape[1] - 1]  # Use the last column
nr_dev = 5  # Threshold for extreme events
extreme_flags = detect_extreme_events(x_normalized, ex_dim, nr_dev)

# Determine which clusters are extreme
for cluster_idx in range(3):
    if np.any(extreme_flags[labels == cluster_idx]):
        cluster_names[cluster_idx] = "Extreme"

# =============================================================================
# Visualize clustering results
# =============================================================================
plt.figure(figsize=(10, 6))
colors = ['#00008B', '#FFEA00', '#ff0000']

for cluster_idx in range(3):
    cluster_points = x_normalized[labels == cluster_idx]
    plt.scatter(
        cluster_points[:, 0],    # First column as x-axis
        cluster_points[:, -1],   # Last column as y-axis
        c=colors[cluster_idx], label=cluster_names[cluster_idx],
        alpha=0.6, edgecolor="black", s=20
    )

# Highlight extreme events explicitly
plt.scatter(
    x_normalized[extreme_flags, 0],
    x_normalized[extreme_flags, -1],
    c='yellow', label="Extreme Events", edgecolor="black", s=50, marker='*'
)

plt.title("Agglomerative Clustering")
plt.xlabel("x_0")
plt.ylabel("Dissipation rate")
plt.legend()
plt.grid(True)
plt.show()


scaler = StandardScaler()
x_normalized = scaler.fit_transform(x_20)

# Create a connectivity matrix for time series
connectivity = kneighbors_graph(x_normalized, n_neighbors=4, mode='connectivity', include_self=False)

# =============================================================================
# Perform Agglomerative Clustering
# =============================================================================
n_clusters = 125  # Define the number of clusters
agg_clustering = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward',
    connectivity=connectivity
)
labels = agg_clustering.fit_predict(x_normalized)

# Cluster naming for visualization
cluster_names = {idx: "Non-Extreme" for idx in range(n_clusters)}

# =============================================================================
# Detect extreme events
# =============================================================================
def detect_extreme_events(x, ex_dim, nr_dev=5):
    """
    Detect extreme events based on a multiple of standard deviations.
    """
    extreme_flags = np.zeros(x.shape[0], dtype=bool)
    for dim in ex_dim:
        mean_val = np.mean(x[:, dim])
        std_dev = np.std(x[:, dim])
        extreme_flags |= abs(x[:, dim]) >= mean_val + nr_dev * std_dev
    return extreme_flags

# Use the last column for extreme event detection
ex_dim = [x_normalized.shape[1] - 1]
nr_dev = 3
extreme_flags = detect_extreme_events(x_normalized, ex_dim, nr_dev)

# Update cluster labels based on extreme events
for cluster_idx in range(n_clusters):
    if np.any(extreme_flags[labels == cluster_idx]):
        cluster_names[cluster_idx] = "Extreme"

# =============================================================================
# Visualize clustering results
# =============================================================================
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
    c='yellow', label="Extreme Events", edgecolor="black", s=100, marker='*'
)

plt.title("Agglomerative Clustering with Connectivity and Extreme Event Detection", fontsize=14)
plt.xlabel("x_0", fontsize=12)
plt.ylabel("Dissipation Rate (x_extreme)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()