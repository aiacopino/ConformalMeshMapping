import numpy as np
import src.operators as ops

class BoundaryCurve:
    def __init__(self, coefficients):
        self.coeffs = np.asarray(coefficients, dtype=complex)
        self.N = len(self.coeffs)
        # print(f"FLAG: Expected 1x2N array for boundary coefficients, got {self.coeffs}.")
        if self.N % 2 != 0:
            raise ValueError("BoundaryCurve __init__: Number of Fourier coefficients must be even.")

    def __len__(self):
        return self.N
    
    def evaluate(self, t):
        '''
        evaluate boundary curve at parameter t (outputs array of complex points on the boundary)
        
        :param t: array of parameter values in [0, 2pi]
        '''
        #result = np.zeros_like(t, dtype=complex)
        t = np.asarray(t)
        k = np.fft.fftfreq(self.N) * self.N
        # result[j] = sum(coeffs[m] * exp(1j * k[m] * t[j]))
#        for i, ti in enumerate(t):
#            result[i] = np.sum(self.coeffs * np.exp(1j * k * ti))
        exponent = 1j * np.outer(t, k)  # t rows, k cols => rank 2 tensor, linalg faster than loop
        return np.dot(np.exp(exponent), self.coeffs) # sum(c_k * e^{ikt}) for each k, t (mat-vec mult)

    def derivative(self):
        '''
        derivative of ^f = ik * coeffs(f) corresponds to swapping real and imag parts in our coeff matrix and mult by i/-i
        '''
        return ops.fourier_derivative(self.coeffs)
    
    def evaluate_derivative(self, t):
        '''
        derivative but in spatial domain
        '''
        deriv_coeffs = self.derivative()
        t = np.asarray(t)
        k = np.fft.fftfreq(self.N) * self.N
        exponent = 1j * np.outer(t, k)  # t rows, k cols
        return np.dot(np.exp(exponent), deriv_coeffs)