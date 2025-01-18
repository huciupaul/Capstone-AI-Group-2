# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:35:33 2023

@author: floris
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

fld = './ExampleData/'
fln = fld + 'Example_data.h5'

hf = h5py.File(fln,'r')
x = np.array(hf.get('Z'))
t = np.array(hf.get('t'))
hf.close()

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
min_clusters = 20
max_it = 1

nr_dev = 1.5 # this is used to decide on what is the definition of extreme events
extr_dim = [x.shape[1]-1] # we assume the extreme dimension is the final one
plotting = False

features = [] # to give names to the features
for i in range(x.shape[1]):
    features.append('x' + str(i))

# =============================================================================
# Some preparatory lists and hyperparameters
# =============================================================================

clusters, D, P, P_graph, tess_ind, tess_ind_trans, tess_ind_cluster = clfunc.extreme_event_identification_process(
        t, x, M, extr_dim, 'Example', min_clusters, max_it, nr_dev, features, plotting, 'classic', False)

# Calculate the statistics of the identified clusters
clfunc.calculate_statistics(extr_dim, clusters, P, tf)

# =============================================================================
# Save the identified clusters
# =============================================================================

fln_r = fld + 'Cluster_'

fln = fln_r + 'D.npz'
sp.save_npz(fln,D)
fln = fln_r + 'P.npz'
sp.save_npz(fln,P)
fln = fln_r + 'P_graph.pickle'
pickle.dump(P_graph,open(fln,'wb'))
fln = fln_r + 'tess_ind_data.h5'
hf = h5py.File(fln,'w')
hf.create_dataset('tess_ind',data=tess_ind)
hf.create_dataset('tess_ind_trans',data=tess_ind_trans)
hf.create_dataset('tess_ind_cluster',data=tess_ind_cluster)
hf.close()

fln = fln_r + 'clusters.h5'
clfunc.save_clusters(fln,clusters)

# =============================================================================
#                  General postprocessing
# =============================================================================

# Check on "new" data series
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
