import numpy as np
from kolsol.numpy.solver import KolSol
import matplotlib.pyplot as plt
import numpy as np

# instantiate solver
ks = KolSol(nk=8, nf=4, re=30.0, ndim=2)
dt = 0.01

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

plt.contour(u_field[:,:,1])
plt.show()