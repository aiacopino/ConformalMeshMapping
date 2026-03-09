import numpy as np
from pathlib import Path
from numpy.fft import fft, ifft
from src.targetregion import BoundaryCurve
import tests.convergence_tests as tests

'''
PARAMETERS

COMMENT

''' 
# file not found error handling
main_dir = Path(__file__).parent

#eta_mat = np.load(main_dir/'data/boundary.npy')
#eta = BoundaryCurve(eta_mat)


# adapt params here for different target regions and convergence tests
N = 512
p = 0.4
relaxation = 0.1
tolerance = 1e-5

# verfahren 1 or 2 from wegmann's paper, 1 is more exact (if it converges) and 2 converges faster. 
# if diverging, try switching to verfahren 2, then lowering relaxation, increasing number of discretisation pts N, or lowering convexity parameter p (if applicable).
# note any value other than 1 will run verfahren 2.
# (note verfahren translates to "method" but we chose the german word because "method"=="fct" in non-py programming; to avoid confusion)
verfahren = 1

# UNCOMMENT THE SHAPE YOU WANT TO TEST BELOW. ordered from easiest to hardest to converge, in general.
#tests.test_unit_disk_convergence(N, verfahren, tolerance)
#tests.test_eccentric_circle_convergence(N, relaxation, verfahren, tolerance)

#initial_guess = "non_convex"
#initial_guess = "starshaped" # for non-convex shapes, identity initial guess can diverge, so we use a more convergence-friendly initial guess. for convex shapes, identity is fine and faster to compute.
#initial_guess = "identity" # for convex shapes, identity is fine and faster to compute. for non-convex shapes, identity initial guess can diverge, so we use a more convergence-friendly initial guess.

#tests.test_inverted_ellipse_convergence(N, p, relaxation, verfahren, initial_guess, tolerance)

#petals = 8
#tests.test_flower_convergence(N, p, relaxation, verfahren, initial_guess, tolerance, petals) #maybe make num of petals adjustable here

#tests.test_kite_convergence(N, relaxation, verfahren, tolerance)

#cutoff here controls the smoothing of the square's edges. cutoff means 1/2 * ratio of coefficients to keep, 
# hence diminish this if diverging to get a rounder shape. cutoff = N//3 -> 2/3 of coeffs are kept
cutoff = 1/100
tests.test_square_convergence(tolerance, N, relaxation, verfahren, cutoff)