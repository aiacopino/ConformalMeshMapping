import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-2, 2, 100)
Z = x + 0j
W = np.sqrt(Z)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Positive real: along +Re (x>0)
ax.plot(x[x>=0], np.zeros_like(x[x>=0]), np.real(W[x>=0]), 'g', label='Re(sqrt(x))')

# Negative real: along +Im (y>0)
ax.plot(np.zeros_like(x[x<0]), -x[x<0], np.imag(W[x<0]), 'r', label='Im(sqrt(x))')

ax.scatter(0, 1, 1, color='red', s=50)  # sqrt(-1)=i

ax.set_xlabel('Re'); ax.set_ylabel('Im'); ax.set_zlabel('sqrt value')
ax.set_title('sqrt on reals: >0 on Re, <0 on Im')
ax.legend()

# origin at back vertex
ax.view_init(elev=20, azim=-150)
ax.xaxis._axinfo['juggled'] = (1,0,2)
ax.yaxis._axinfo['juggled'] = (0,1,2)
ax.zaxis._axinfo['juggled'] = (2,0,1)

plt.show()