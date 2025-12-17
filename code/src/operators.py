import numpy as np
## math helpers

## hilbert transform
## https://www.youtube.com/watch?v=dy4OeAYqSqM
def hilbert_transform(matrix):
    '''
    param: matrix 2xN of fourier coefficients
    returns: matrix 2xN of fourier coefficients of hilbert transform

    in time domain:
    # set all negative frequency coefficients to zero
    # double the real coefficients
    # IFFT
    # imaginary part is the hilbert transform 
    in frequency domain:
    # swap real & imaginary coefficients
    # multiply imaginary coefficients by -1
    # set k=o frquency coefficient = 0
    '''
    real_part = matrix[0,:]
    imag_part = matrix[1,:]

    matrix_hilbert = np.zeros_like(matrix)
    matrix_hilbert[0,:] = imag_part
    matrix_hilbert[1,:] = -real_part
    matrix_hilbert[:,0] = 0 # zero frequency component

    return matrix_hilbert

## derivative
def fourier_derivative():
    return None

## cauchy integral formula to compute the region's interior mesh from the boundary once mapped