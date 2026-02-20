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
p = 0.3
relaxation = 0.01
tolerance = 1e-10

# verfahren 1 or 2 from wegmann's paper, 1 is more exact (if it converges) and 2 converges faster. 
# if diverging, try switching to verfahren 2, then lowering relaxation, increasing number of discretisation pts N, or lowering convexity parameter p (if applicable).
# note any value other than 1 will run verfahren 2.
# (note verfahren translates to "method" but we chose the german word because "method"=="fct" in non-py programming; to avoid confusion)
verfahren = 2

# UNCOMMENT THE SHAPE YOU WANT TO TEST BELOW. ordered from easiest to hardest to converge, in general.
#tests.test_unit_disk_convergence(N, verfahren)
#tests.test_eccentric_circle_convergence(N, relaxation, verfahren, tolerance)
tests.test_inverted_ellipse_convergence(N, p, relaxation, verfahren) #converges for relaxation >0.8
#tests.test_flower_convergence(N, p, relaxation, verfahren) #maybe make num of petals adjustable here
#tests.test_kite_convergence(N, relaxation, verfahren)

#cutoff here controls the smoothing of the square's edges. cutoff means 1/2 * ratio of coefficients to keep, 
# hence diminish this if diverging to get a rounder shape. cutoff = N//3 -> 2/3 of coeffs are kept
#cutoff = 1/6
#tests.test_square_convergence(tolerance, N, relaxation, verfahren, cutoff)