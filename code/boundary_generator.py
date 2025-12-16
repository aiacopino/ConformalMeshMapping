import numpy as np
import os

N = 5
eta = np.zeros((2, N))

#circle with perturbation
eta[0,1]=1
eta[0,3]=0.1

print(eta)

file_path = 'data/boundary.npy'
np.save(file_path, eta)
print("saved boundary data to ", os.path.abspath(file_path))