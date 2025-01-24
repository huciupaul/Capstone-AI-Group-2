import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import h5py

# =============================================================================
# Read the data
# =============================================================================

# fld = './ExampleData/'
# fln = fld + '48_Encoded_data_Re40_8_18_1.h5'
fln = r"C:\Users\agata\Downloads\48_Encoded_data_Re40_12_22_1.h5"

hf = h5py.File(fln, 'r')
x = np.array(hf.get('U_enc'))

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

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=1, min_samples=2)  # eps is the maximum distance between two samples to be considered as in the same neighborhood
labels = dbscan.fit_predict(x_normalized)

# Cluster naming for visualization
cluster_names = {label: "Normal" for label in set(labels) if label != -1}  # -1 is used for noise

# =============================================================================
# Detect extreme events based on standard deviation of dissipation rate
# =============================================================================
def detect_extreme_events(x, ex_dim, nr_dev=5):
    extreme_flags = np.zeros(x.shape[0], dtype=bool)
    for dim in ex_dim:
        mean_val = np.mean(x[:, dim])
        std_dev = np.std(x[:, dim])
        extreme_flags |= abs(x[:, dim]) >= mean_val + nr_dev * std_dev
    return extreme_flags

ex_dim = [x_normalized.shape[1] - 1]  # Use the last column as dissipation rate
nr_dev = 3.75  # Threshold for extreme events
extreme_flags = detect_extreme_events(x_normalized, ex_dim, nr_dev)

# =============================================================================
# Identify pre-cursor clusters (clusters that are sequentially before an extreme cluster)
# =============================================================================
# Find the indices of extreme events
extreme_indices = np.where(extreme_flags)[0]

# Identify pre-cursor clusters (sequential clusters before extreme ones)
precursor_flags = np.zeros(len(x_normalized), dtype=bool)
for idx in extreme_indices:
    if idx > 0:  # Ensure we don't go out of bounds
        precursor_flags[idx - 1] = True  # Mark the previous point as a precursor

# Update cluster names for clusters containing extreme events or pre-cursor events
for cluster_idx in set(labels):
    if cluster_idx != -1:  # Skip noise points
        cluster_points = x_normalized[labels == cluster_idx]
        
        # If any point in the cluster is an extreme event, mark it as "Extreme"
        if np.any(extreme_flags[labels == cluster_idx]):
            cluster_names[cluster_idx] = "Extreme"
        # If any point in the cluster is a precursor event, mark it as "Pre-cursor"
        elif np.any(precursor_flags[labels == cluster_idx]):
            cluster_names[cluster_idx] = "Pre-cursor"

# =============================================================================
# Visualize clustering results
# =============================================================================
plt.figure(figsize=(10, 6))

# Set colors for the different cluster types
colors = {'Extreme': '#ff0000', 'Pre-cursor': '#FFEA00', 'Normal': '#0000FF'}

# Plot all data points and color them based on their cluster
for cluster_idx in set(labels):
    if cluster_idx != -1:  # Skip noise points
        cluster_points = x_normalized[labels == cluster_idx]
        # Assign color based on the cluster type
        cluster_color = colors.get(cluster_names.get(cluster_idx, "Normal"), '#0000FF')
        # Plot all points in the cluster with the assigned color
        plt.scatter(
            cluster_points[:, 0],    # First column as x-axis
            cluster_points[:, -1],   # Last column as y-axis (dissipation rate)
            c=cluster_color, alpha=0.6, edgecolor="black", s=20
        )

# Highlight extreme events explicitly with a different marker
plt.scatter(
    x_normalized[extreme_flags, 0],
    x_normalized[extreme_flags, -1],
    c='yellow', label="Extreme Events", edgecolor="black", s=50, marker='*'
)

# Manually add legend entries
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff0000', markersize=10, label='Extreme'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFEA00', markersize=10, label='Pre-cursor'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#0000FF', markersize=10, label='Normal'),
    Line2D([0], [0], marker='*', color='w', markersize=10, markerfacecolor='yellow', markeredgecolor="black", label='Extreme Events')
]

# Add the manual legend to the plot
plt.legend(handles=legend_elements)
plt.title("DBSCAN Clustering")
plt.xlabel("x_0")
plt.ylabel("Dissipation rate")
plt.grid(True)
plt.show()
