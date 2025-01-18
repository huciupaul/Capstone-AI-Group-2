
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

fln = r"C:\Users\agata\Downloads\48_Encoded_data_Re40_8_18_1.h5"

print(os.getcwd())

hf = h5py.File(fln, 'r')
x = np.array(hf.get('U_enc'))

print(f"Shape of x: {x.shape}")

# Use only 20% of the data
num_elem = int(len(x) * 0.2)
x_20 = x[:num_elem]

scaler = StandardScaler()
x_normalized = scaler.fit_transform(x_20)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=400)
labels = kmeans.fit_predict(x_normalized)

# Map cluster labels to names
cluster_names = {0: "Normal", 1: "Precursor", 2: "Extreme"}
named_clusters = [cluster_names[label] for label in labels]

# Scatter plot using first and last columns for axes
plt.figure(figsize=(10, 6))
colors = ['blue', 'orange', 'red']

for label, color, name in zip(range(3), colors, cluster_names.values()):
    cluster_points = x_normalized[labels == label]
    plt.scatter(
        cluster_points[:, 0],    # First column as x-axis
        cluster_points[:, -1],   # Last column as y-axis
        c=color, label=name, alpha=0.6, edgecolor="black", s=20
    )

plt.title("K-Means Clustering")
plt.xlabel("x_0")
plt.ylabel("x_extreme")
plt.legend()
plt.grid(True)
plt.show()


