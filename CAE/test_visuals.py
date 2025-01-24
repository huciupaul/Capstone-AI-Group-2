from visualization import plot_training_curve, read_mse_plot, plot_hyperparameter_tuning

vloss_plot, tloss_plot = read_mse_plot("mse_plot_10_100.h5")
plot_training_curve(vloss_plot, tloss_plot, 50, n_lat=10)

# plot_hyperparameter_tuning('hyperparameter_tuning.txt')
