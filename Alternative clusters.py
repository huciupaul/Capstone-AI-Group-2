
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
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

hf = h5py.File(fln,'r')
x = np.array(hf.get('U_enc'))

num_elem = int(len(x) + 0.2)  # Use only 20% of the data 
x_20 = x[:num_elem]

x_reshaped = x_20.reshape(-1, 1)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(x_reshaped)

cluster_names = {0: "Normal", 1: "Precursor", 2: "Extreme"}
named_clusters = [cluster_names[label] for label in labels]

# Visualize clustering results
plt.figure(figsize=(10, 6))
for label, color, name in zip(range(3), ['blue', 'orange', 'red'], cluster_names.values()):
    cluster_points = x_reshaped[np.array(labels) == label]
    plt.scatter(cluster_points, [label] * len(cluster_points), c=color, label=name, alpha=0.6)

plt.title("K-Means Clustering of Dissipation Rate")
plt.xlabel("Dissipation Rate")
plt.ylabel("Cluster")
plt.legend()
plt.show()