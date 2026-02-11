import numpy as np
#from scipy.fft import fft, ifft
## math helpers

## https://www.youtube.com/watch?v=dy4OeAYqSqM
## hilbert tranform
# input: spatial domain values
# output: spatial domain values
def hilbert_transform(v_val):
    '''
    n = len(vals)
    assert n % 2 == 0, "N must be even for clean FFT behavior"
    v_hat = np.fft.fft(vals)

    h_hat = np.zeros(n, dtype=complex)
    half = n // 2
    h_hat[1:half]   = -1j * v_hat[1:half]     # positive frequencies
    h_hat[half+1:]  =  1j * v_hat[half+1:]    # negative frequencies
    # h_hat[0] = h_hat[half] = 0  (already zero)

    return np.fft.ifft(h_hat).real
    '''
    N = len(v_val)
    v_hat = np.fft.fft(v_val)
    h_hat = np.zeros(N, dtype=complex)
    
    # H[u] multiplier is -i * sgn(k)
    # Frequencies: 0, 1..N/2-1, N/2, -N/2+1..-1
    #k=0 zeroed out since HT of const is 0
    half = N // 2
    h_hat[1:half] = -1j * v_hat[1:half]       # + freq
    h_hat[half+1:] = 1j * v_hat[half+1:]      # - freq
    
    return np.fft.ifft(h_hat).real


## derivative of a fourier coeff array
def fourier_derivative(coeffs_hat):
    N = len(coeffs_hat)
    k = np.fft.fftfreq(N) * N
    return 1j * k * coeffs_hat

## cauchy integral formula to compute the region's interior mesh from the boundary once mapped