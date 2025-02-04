import numpy as np
import matplotlib.pyplot as plt
import h5py
from constants import N_x, N_y

def plot_physical_interpretation(U_test, N_lat, centroid_type, save_path):
    """
    Plots the decoded centroid data in a grid format (5x2) and saves it to a PDF.

    Args:
        U_test: Decoded velocity field data (array of shape (num_centroids, N_x, N_y, channels)).
        N_lat: Number of latent dimensions (for labeling purposes).
        centroid_type: Type of centroids (e.g., "Precursor", "Extreme", "Normal").
        save_path: Path to save the output PDF.
    """
    # Grid for the velocity field
    X = np.linspace(0, 2 * np.pi, N_x) 
    Y = np.linspace(0, 2 * np.pi, N_y) 
    XX = np.meshgrid(X, Y, indexing='ij')

    # Set fixed scale range for all plots
    vmin, vmax = -2.4, 2.4  # Fixed color scale for all plots

    # Plot settings
    num_centroids = U_test.shape[0]  # Number of centroids
    grid_rows = 2  # Fixed number of rows
    grid_cols = 5  # Fixed number of columns
    plt.rcParams["figure.figsize"] = (15, 6)  # Adjust figure size
    plt.rcParams["font.size"] = 12

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 6))
    axes = axes.flatten()  # Flatten for easy iteration

    for i in range(num_centroids):
        ax = axes[i]
        u = U_test[i, :, :, 1]  # Take the first channel (e.g., u-velocity)
        
        # Plot filled contours with fixed scale
        contourf = ax.contourf(XX[0], XX[1], u, levels=10, cmap='coolwarm', vmin=vmin, vmax=vmax)
        
        # Overlay line contours
        ax.contour(XX[0], XX[1], u, levels=10, colors='black', linewidths=0.5)

        # Add colorbar **for each plot**, with the **same scale** (-2.8 to 2.8)
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
        cbar.set_ticks(np.linspace(vmin, vmax, num=5))  # Ensure tick labels are consistent
        cbar.ax.tick_params(labelsize=8)  # Reduce font size for better fit

        # Set the title for each subplot
        ax.set_title(f"Centroid {i + 1} - Y", fontsize=10)
        ax.axis('off')  # Turn off axes for a cleaner look

    # Remove unused axes if there are fewer centroids
    for i in range(num_centroids, len(axes)):
        fig.delaxes(axes[i])

    # Set the main title for the entire plot
    fig.suptitle(f"Decoded Velocity Fields - {centroid_type} Centroids", fontsize=16, y=0.98)

    # Save the figure to a PDF
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    plt.savefig(save_path)
    plt.show()


# Path to the HDF5 file (comment/uncomment based on the centroids to plot)
#path = r'.\Data\48_Decoded_data_Re40_10_Precursor_Centroids.h5'
#path = r'.\Data\48_Decoded_data_Re40_10_Extreme_Centroids.h5'
path = r'.\Data\48_Decoded_data_Re40_10_Normal_Centroids.h5'

# Extract data from the HDF5 file
with h5py.File(path, 'r') as hf:
    print("Keys in HDF5 file:", list(hf.keys()))
    U = np.array(hf.get('U_dec'), dtype=np.float32)

# Extract the correct centroid type from the path
centroid_type = path.split('_')[-2]  

# Plot the centroids
save_path = f"./Decoded_{centroid_type}_Centroids.pdf"  # Save file with centroid type
plot_physical_interpretation(
    U_test=U,
    N_lat=10,  # Latent space dimension
    centroid_type=centroid_type,  # Type of centroid
    save_path=save_path  # Path to save the output PDF
)
