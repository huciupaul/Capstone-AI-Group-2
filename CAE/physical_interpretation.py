import numpy as np
import matplotlib.pyplot as plt
import textwrap
import h5py
from constants import N_x, N_y

# Function to wrap text for pretrier display of text in subplots
def wrap_text(text, width=25):
    return "\n".join(textwrap.wrap(text, width))


def plot_physical_interpretation(U_test, N_lat):

    # grid
    X = np.linspace(0, 2 * np.pi, N_x) 
    Y = np.linspace(0, 2 * np.pi, N_y) 
    XX = np.meshgrid(X, Y, indexing='ij')

    # plot n snapshots and their reconstruction in the test set.
    n_snapshots = 5
    plt.rcParams["figure.figsize"] = (15, 4 * n_snapshots)
    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(n_snapshots, 3)
    
    for i in range(n_snapshots):
        print(f"Plotting snapshot {i+1}/{n_snapshots}")
        # testing data
        u = U_test.copy()
        vmax = u.max()
        vmin = u.min()
        
        # truth
        ax_truth = plt.subplot(n_snapshots, 3, i * 3 + 1)
        CS0 = ax_truth.contourf(XX[0], XX[1], u[0, :, :, 0],
                                levels=10, cmap='coolwarm', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(CS0, ax=ax_truth)
        CS = ax_truth.contour(XX[0], XX[1], u[0, :, :, 0],
                            levels=10, colors='black', linewidths=.5, linestyles='solid',
                            vmin=vmin, vmax=vmax)
        title = wrap_text(f'True velocity field at snapshot {i+1}')
        ax_truth.set_title(title, pad=20, fontsize=12)
    
    # Adjust spacing between plots
    fig.tight_layout(pad=1.0)  # Increase the padding between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Add extra spacing between rows and columns

    # path = './Data/48_RE40_'+str(N_lat)
    plt.savefig(f'./physical_interpretation_{N_lat}.pdf')
    plt.show()

path = r'.\Data\48_Decoded_data_Re40_10_Precursor_Centroids.h5'
#path = r'.\Data\48_Decoded_data_Re40_10_Extreme_Centroids.h5'
#path = r'.\Data\48_Decoded_data_Re40_10_Normal_Centroids.h5'

with h5py.File(path, 'r') as hf:
    print(list(hf.keys()))
    U = np.array(hf.get('U_dec'), dtype=np.float32)

plot_physical_interpretation(U, 10)