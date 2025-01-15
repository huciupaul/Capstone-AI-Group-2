import os
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np




def plot_training_curve(vloss_plot, tloss_plot, N_check, epoch):
    plt.rcParams["figure.figsize"] = (15,4)
    plt.rcParams["font.size"]  = 20
    plt.title('MSE convergence')
    plt.yscale('log')
    plt.grid(True, axis="both", which='both', ls="-", alpha=0.3)
    plt.plot(tloss_plot[np.nonzero(tloss_plot)], 'y', label='Train loss')
    plt.plot(np.arange(np.nonzero(vloss_plot)[0].shape[0])*N_check,
                vloss_plot[np.nonzero(vloss_plot)], label='Val loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'MSE{epoch}.pdf')
    plt.show()