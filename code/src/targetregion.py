import numpy as np
class BoundaryCurve:
    def __init__(self, matrix):
        rows, cols = matrix.shape
        self.coeffs = matrix
        self.N = cols

        if rows != 2:
            raise ValueError(f"Expected 2xN array for boundary coefficients, got {rows}x{cols}.")

    def evaluate(self, t):
        pass

    def derivative(self): #derivative of ^f = ik * coeffs(f) corresponds to swapping real and imag parts in our coeff matrix and mult by i/-i
        k_vec = np.arange(self.N)
        deriv_coeffs = np.zeros_like(self.coeffs)
        deriv_coeffs[0,:] = -k_vec * self.coeffs[1,:] 
        deriv_coeffs[1,:] = k_vec * self.coeffs[0,:]
        return deriv_coeffs