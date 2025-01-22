#WARNING: the data file is too big for GitHub. DO NOT commit any cahnges to GitHub - it will give bunch of errors
# the data file path: TEAMS -> DOCUMENTS -> DATA FROM CODING -> GENERATED_DATA.H5


import numpy as np
from kolsol.numpy.solver import KolSol
import matplotlib.pyplot as plt
import numpy as np
import h5py
import tqdm
import time


np.random.seed(42)


# instantiate solver
ks = KolSol(nk=8, nf=4, re=40.0, ndim=2) # nk = no of symmetric wavenumbers   nf = forcing frequency, re=40 fully turbulent flow
dt = 0.01
datalen = 30000
steps = int(datalen / dt)
store_interval = 20  # Store every 50 steps
n_stored = steps // store_interval
# the timestep to store the resuls is 0.2


# define initial conditions
u_hat = ks.random_field(magnitude=10.0, sigma=2.0, k_offset=[0, 3])


# simulate :: run over transients to stabilize
for _ in np.arange(0.0, 200.0, dt):
  u_hat += dt * ks.dynamics(u_hat)


# data initialization
K = np.zeros(n_stored)
D = np.zeros(n_stored)
timer = np.zeros(n_stored)
u_field_all = np.zeros((n_stored, 48, 48, 2))  # Assuming u_field is (48, 48, 2)


# Simulate :: generate results
start_time = time.time()

store_idx = 0
for step, t in enumerate(np.arange(0.0, datalen, dt)):
    u_hat += dt * ks.dynamics(u_hat)
    if step % store_interval == 0:  # Store every 10 steps
        timer[store_idx] = t
        D[store_idx] = ks.dissip(u_hat)
        K[store_idx] = 0.5 * np.sum(np.abs(u_hat)**2)
        u_field_all[store_idx] = ks.fourier_to_phys(u_hat, nref=48)  # Compute physical field
        store_idx += 1

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Simulation loop execution time: {elapsed_time:.2f} seconds")

filename = f"Generated_data_{n_stored}.h5"
def write_h5(velocity: np.ndarray, kinetic_energy: np.ndarray, dissipation_rates: np.ndarray) -> None:
    with h5py.File(filename, 'w') as hf:
        # Save velocity as a dataset
        hf.create_dataset("velocity_field", data=velocity, compression="gzip")

        # Save kinetic energy list
        hf.create_dataset("kinetic_energy", data=kinetic_energy, compression="gzip")

        # Save dissipation rates list
        hf.create_dataset("dissipation_rate", data=dissipation_rates, compression="gzip")
    print(f"Results saved to Generated_data.h5")
  
write_h5(u_field_all, K, D)


# Verify the shapes of the stored arrays
print(f"D shape: {D.shape}, K shape: {K.shape}, u_field_all shape: {u_field_all.shape}")
    

with h5py.File(filename, "r") as h5file:
    velocity = h5file["velocity_field"][:]
    dissipation = h5file["dissipation_rate"][:]
    kinetic_energy = h5file["kinetic_energy"][:]


# Plot the dissipation rate
plt.figure(figsize=(15, 6))
plt.plot(timer, dissipation, label="Dissipation Rate")
plt.xlabel("Steps")
plt.ylabel("Magnitude")
plt.legend()
plt.title("Dissipation Rate Over Time (Every 10 Steps)")
plt.savefig('Dissip_plot.pdf', dpi=500, bbox_inches='tight')  # Save as a PDF file
plt.show()

# Plot the kinetic energy
plt.figure(figsize=(15, 6))
plt.plot(timer, kinetic_energy, label="Kinetic energy")
plt.xlabel("Steps")
plt.ylabel("Magnitude")
plt.legend()
plt.title("Kinetic energy Over Time (Every 10 Steps)")
plt.savefig('KE_plot.pdf', dpi=500, bbox_inches='tight')  # Save as a PDF file
plt.show()