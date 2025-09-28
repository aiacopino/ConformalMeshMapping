# aim: visualizing the effect of crowding as described in WEG p. 359
# It was first noted by Grassman [86] that the numerical calculation of the mapping from the disk to an elongated region becomes laborious due to an effect which is now called crowding. The images of equidistributed points on the unit circle become very unevenly distributed on the boundary of the region.

import numpy as np
import matplotlib.pyplot as plt

def phi(z,r):
    return np.arctanh(r*z)

# r_values = [0.1, 0.5, 0.9, 0.99]
r = 0.99

# unit disk grid
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X,Y = np.meshgrid(x, y)
Z = X + 1j*Y

# mask bc phi is only def on unit disk
mask = np.abs(Z) <= 1
X, Y = X[mask], Y[mask]
Z = Z[mask]

#equidistr pts on unit circle
theta = np.linspace(0, 2*np.pi, 100)
z_on_disk = np.exp(1j*theta)
z_mapped = phi(z_on_disk, r)

#both plots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f'Crowding effect for r={r}', fontsize=16)

#left plot
phi_plot1 = ax1.scatter(X, Y, c=phi(Z,r).real, cmap='RdBu', alpha=0.5) # alpha = transparency
fig.colorbar(phi_plot1, ax=ax1, label='Re(Φ(z))')
ax1.axhline(0, color='k', linewidth=0.5)
ax1.axvline(0, color='k', linewidth=0.5)
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_aspect('equal')
ax1.set_title('Unit Disk Mapping')
ax1.set_xlabel('Re(z)')
ax1.set_ylabel('Im(z)')

# right plot
ax2.plot(z_mapped.real, z_mapped.imag, 'k--', linewidth = 2)
ax2.scatter(z_mapped.real, z_mapped.imag, c='k', marker = '*', s=20)
ax2.axhline(0, color='k', linewidth=0.5)
ax2.axvline(0, color='k', linewidth=0.5)
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-1.2, 1.2)
ax2.set_aspect('equal')
ax2.set_title('Equidist. pts on unit circle mapped under phi')
ax2.set_xlabel('Re(z)')
ax2.set_ylabel('Im(z)')
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()
# plt.figure(figsize=(10, 8))
# # for r in r_values:
# #    plt.plot(z, phi(z, r).real, label=f'r={r}')
# plt.scatter(X, Y, c=phi(Z,r).real, cmap='RdBu', alpha=0.5)
# plt.colorbar(label='Re(Φ(z))')
#
# plt.xlabel('Re(z)')
# plt.ylabel('Im(z)')
# plt.title('Effect of crowding in elongated regions')
# plt.legend()
# plt.grid()
# plt.show()
