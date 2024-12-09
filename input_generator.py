import numpy as np
from kolsol.numpy.solver import KolSol
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# instantiate solver
ks = KolSol(nk=8, nf=4, re=40.0, ndim=2) # nk = no of symmetric wavenumbers   nf = forcing frequency
dt = 0.01
# the timestep to store the resuls is 0.1

# define initial conditions
u_hat = ks.random_field(magnitude=10.0, sigma=2.0, k_offset=[0, 3])

# simulate :: run over transients
for _ in np.arange(0.0, 180.0, dt):
  u_hat += dt * ks.dynamics(u_hat)

# simulate :: generate results
for _ in np.arange(0.0, 300.0, dt):
  u_hat += dt * ks.dynamics(u_hat)

# generate physical field
u_field = ks.fourier_to_phys(u_hat, nref=256)

#plt.contour(u_field[:,:,0])
#plt.show()
print(u_field.shape)

#help(KolSol)