from visualization import plot_training_curve, read_mse_plot

vloss_plot, tloss_plot = read_mse_plot("mse_plot_40.h5")
plot_training_curve(vloss_plot, tloss_plot, 50)