from visualization import illustrate_autoencoder, plot_training_curve, read_mse_plot, plot_hyperparameter_tuning



# plot and save training curve
#n_lat = 10
#epochs = 120
#vloss_plot, tloss_plot = read_mse_plot(f"mse_plot_{n_lat}_{epochs}.h5")
#plot_training_curve(vloss_plot, tloss_plot, epoch=epochs, n_lat=n_lat)

# plot hyperpameter tuning
plot_hyperparameter_tuning("./Plots/Tuning/hyperparameter_tuning.txt")





