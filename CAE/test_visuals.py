from visualization import plot_training_curve
import numpy as np

vloss_plot = np.array([0.29476211, 0.15889723, 0.12333059, 0.10619613, 0.09521885, 0.09354889, 0.09357198, 0.08833057, 0.0906551 , 0.08948215, 0.09024367])
tloss_plot = np.array([0.35731878, 0.22195849, 0.14154739, 0.11547023, 0.1005634 , 0.09252081, 0.0868399 , 0.08161513, 0.07679051, 0.07258764, 0.06888187])

plot_training_curve(vloss_plot, tloss_plot, 1, 1)