
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg

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

fln = r"C:\Users\agata\Downloads\Generated_data_96000.h5"


import os
print(os.getcwd())

hf = h5py.File(fln,'r')
x = np.array(hf.get('dissipation_rate'))

print(x)