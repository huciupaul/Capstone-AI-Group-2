# -*- coding: utf-8 -*-
"""

This is a script that illustrates how to load a dataset and analyse it using already identified precursor clusters

@author: akdoan
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import clustering_func_only_features as clfunc
import pandas as pd
from scipy.integrate import simps
import scipy.sparse as sp
import time
import scipy.stats
import h5py
import pickle

start_time = time.time()


# =============================================================================
# Read the data
# =============================================================================

fld = "./Data/"
fln = fld + "48_Encoded_data_Re40_8_18_1.h5"

hf = h5py.File(fln,'r')
x = np.array(hf.get('U_enc'))
t = np.array(hf.get('t'))
hf.close()

x = x[:2000,:]
t = t[:2000]
dt = t[1]-t[0] # time step
tf = t[-1] # final time
Nt = x.shape[0] # first dimension of x is time
Ndim = x.shape[1] # second dimension is the list of elements
# The index through which extreme events are measured is the last (i.e., x[:,-1] is the equivalent of your dissipation rate)

# normalization
x = (x - np.mean(x,0).reshape((1,Ndim)) ) / np.std(x,0).reshape((1,Ndim))


# =============================================================================
# Some preparatory lists and hyperparameters
# =============================================================================

prediction_time_list = []
false_positive_list = []
false_negative_list = []
instances_list = []


# Provide the hyperparameter for the clustering
M = 15  #Number of tessellation sections per phase space dimension
nr_dev = 1.5 # this is used to decide on what is the definition of extreme events

extr_dim = [x.shape[1]-1] # we assume the extreme dimension is the final one

features = [] # to give names to the features
for i in range(x.shape[1]):
    features.append('x' + str(i))

# =============================================================================
# Loading the results of a previous cluster
# =============================================================================

fln_r = fld + 'Cluster_'

fln = fln_r + 'D.npz'
D = sp.load_npz(fln)

fln = fln_r + 'P.npz'
P = sp.load_npz(fln)

fln = fln_r + 'tess_ind_data.h5'
hf = h5py.File(fln,'r')
tess_ind = np.array(hf.get('tess_ind'))
tess_ind_trans = np.array(hf.get('tess_ind_trans'))
tess_ind_cluster = np.array(hf.get('tess_ind_cluster'))
hf.close()

fln = fln_r + 'clusters.h5'
clusters, extr_clusters, prec_clusters = clfunc.load_clusters(fln,P)


# =============================================================================
#                  General postprocessing
# =============================================================================

# Check on "new" data series (the array x)
# Here we take the old data series and feed it to the algorithm as if it was new
# Tessellate data set (without extreme event identification)
x_tess, temp = clfunc.tesselate(x, M, extr_dim, nr_dev)
x_tess = clfunc.tess_to_lexi(x_tess, M, x.shape[1])

# Translate data set to already identified clusters
x_clusters = clfunc.data_to_clusters(x_tess, D, x, clusters)

is_extreme = np.zeros_like(x_clusters)
for cluster in clusters:
        # New data series, determining whether the current
    is_extreme[np.where(x_clusters == cluster.nr)] = cluster.is_extreme
        # state of the system is extreme (2), precursor (1) or normal state (0)

# =============================================================================
#                  Analysis of the statistics of the clusters
# =============================================================================

print("These are the features: ", features)
# Calculate the false positive and false negative rates
avg_time, instances, instances_extreme_no_precursor, instances_precursor_no_extreme, instances_precursor_after_extreme = clfunc.backwards_avg_time_to_extreme(
        is_extreme, dt)
print('Average time from precursor to extreme:', avg_time, ' seconds')
print('Nr times when extreme event had a precursor:', instances)
print('Nr extreme events without precursors (false negative):',
          instances_extreme_no_precursor)
print('Percentage of false negatives:', instances_extreme_no_precursor /
          (instances+instances_extreme_no_precursor)*100, ' %')
print('Percentage of extreme events with precursor (correct positives):',
          instances/(instances+instances_extreme_no_precursor)*100, ' %')
print('Nr precursors without a following extreme event (false positives):',
          instances_precursor_no_extreme)
print('Percentage of false positives:', instances_precursor_no_extreme /
          (instances+instances_precursor_no_extreme)*100, ' %')

prediction_time_list.append(avg_time)
false_negative_list.append(instances_extreme_no_precursor)
false_positive_list.append(instances_precursor_no_extreme)
instances_list.append(instances)

print("--- %s seconds ---" % (time.time() - start_time))

# =============================================================================
# Phase space plot
# =============================================================================

zindex = 0 # index of dimension in latent space
features_data = x.copy()

plt.figure(figsize = (6,6))
plt.plot(features_data[:,zindex], features_data[:,-1])
plt.grid()
plt.xlabel("$x$" + str(zindex))
plt.ylabel("$x_extreme$")


# =============================================================================
# Tesselated phase space plots
# =============================================================================
palette = plt.get_cmap('viridis', D.shape[1])
coord_clust_centers, coord_clust_centers_tess = clfunc.cluster_centers(x,tess_ind, tess_ind_cluster, D, x.shape[1])

# the plotting function below will plot the clustered phase space (one color per cluster) of (x[:,0], x[:,-1]) (so first dimension and extreme dimension)
# you should adapt it for your own plotting
clfunc.plot_phase_space_clustered(x, 'Example', D, tess_ind_cluster, coord_clust_centers, extr_clusters, prec_clusters, 1.5, palette, features)
plt.show()


# =============================================================================
# Dissipation time series with background color plot
# =============================================================================

t_plot = t
changes_timestep = []
for i in range(len(is_extreme)-1):
    if is_extreme[i] != is_extreme[i + 1]:
        changes_timestep.append(i)

changes_timestep.insert(0, 0)
last_index = len(changes_timestep)
changes_timestep.insert(last_index, len(t_plot)-1)

fig, ax1 = plt.subplots(figsize=(8, 3))

color1 = 'k'
ax1.set_xlabel("t", fontsize=17)
# ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax1.tick_params(axis='x', labelcolor=color1, labelsize=17)
ax1.set_ylabel(r'x_extreme', color=color1, fontsize=17)

ax1.plot(t_plot, features_data[:, -1],  color=color1, linewidth=1.0)

ax1.tick_params(axis='y', labelcolor=color1, labelsize=17)

facecolors = ['skyblue', 'moccasin', 'salmon']
for i in range(len(changes_timestep)-1):
    plt.axvspan(t_plot[changes_timestep[i]], t_plot[changes_timestep[i+1]],
                facecolor=facecolors[is_extreme[changes_timestep[i+1]]],
                alpha=0.5, zorder=-100)
    plt.axvline(x=t_plot[changes_timestep[i+1]],
                linestyle='--', color='tab:red', linewidth=1.0)
    # Change this line to crop the image

plt.xlim(t_plot[0], t_plot[-1])

plt.show()

