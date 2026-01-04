import numpy as np
import src.operators as ops

class BoundaryCurve:
    def __init__(self, coefficients):
        self.coeffs = np.asarray(coefficients, dtype=complex)
        self.N = len(self.coeffs)
        print(f"FLAG: Expected 1x2N array for boundary coefficients, got {self.coeffs}.")
        if self.N % 2 != 0:
            raise ValueError("BoundaryCurve __init__: Number of Fourier coefficients must be even.")

    def evaluate(self, t):
        #evaluate boundary curve at parameter t (array of values in [0, 2pi])
        #using fourier series
        N = self.N
        result = np.zeros_like(t, dtype=complex)
        k = np.fft.fftfreq(N) * N
        for i in enumerate(t): # THIS NEEDS T TO BE A VECTOR
            result[i] = np.sum(self.coeffs * np.exp(1j * k * t))
#        for i, ti in enumerate(t):
#            result[i] = np.sum(self.coeffs * np.exp(1j * k * ti)) WHY WOULD ONE WRITE THIS
        return result

    def derivative(self): #derivative of ^f = ik * coeffs(f) corresponds to swapping real and imag parts in our coeff matrix and mult by i/-i
        return ops.fourier_derivative(self.coeffs)
    
    '''
    def boundaryCorrespondence():
        #S: [0, 2pi] -> [0, 2pi], theta \mapsto s mapping angle on unit disk to boundary parameter of target region eta(s). diffeo of the circle -> not periodic
        # S(theta)= theta + u(theta) where u is 2pi-periodic smooth


        return None
    '''