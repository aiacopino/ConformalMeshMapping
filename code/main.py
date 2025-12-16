import numpy as np
from numpy.fft import fft, ifft
from src.targetregion import BoundaryCurve

'''
PARAMETERS
boundary matrix 2xN containing fourier coefficients of the boundary function of the target region Omega
mesh matrix 2xN containing coordinates of the vertices of the mesh on the domain (unit disk)
mesh matrix 3xM containing indices of the triangles of the mesh
RETURNS 
matrix 2xN containing coordinates of the vertices of the mesh on the target region Omega

''' 

eta_mat = np.load('data/boundary.npy')
eta = BoundaryCurve(eta_mat)
