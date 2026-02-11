import numpy as np
import matplotlib.pyplot as plt
from src.wegmann import WegmannSolver
from src.targetregion import BoundaryCurve

'''
in this file we test the algorithm as a whole 
to ensure convergence for the most common target region shapes
'''

class UnitCircle:
    def eval(self,t):
        return np.exp(1j*t)
    
def test_omega_unit_circle():
    '''
    expected immediate convergence for unit circle as target region
    '''
    print("testing wegmann solver for unit circle target region ...")
    N=128
    eta_circle = UnitCircle()
    eta_coeffs = np.fft.fft(eta_circle.eval(np.linspace(0, 2*np.pi, N, endpoint=False)))/N #discretisation
    circle_bdary_obj = BoundaryCurve(eta_coeffs) #make bdarycurve object from unit circle
    solver = WegmannSolver(circle_bdary_obj, N)
    print("Running wegmann solver for unit circle target region ...")
    res_sigma = solver.find_conformal_map(max_iter=10, epsilon=1e-10)
    
    max_sigma = np.max(np.abs(res_sigma))
    print(f"Max perturbation (sigma) magnitude: {max_sigma}")
    
    if max_sigma < 1e-10:
        print("SUCCESS: Sigma is effectively zero (Identity Map preserved).")
    else:
        print("FAILURE: Sigma should be zero for identity map.")

    # 2. Check the Error History
    # The first error calculation might be non-zero depending on implementation details,
    # but it should drop to machine precision immediately.
    print("Error History:", solver.error_history)
    
    # 3. Geometric Check
    # Let's see if the final points f_k actually lie on the circle
    # We manually reconstruct the boundary points
    sigma_val = np.fft.ifft(solver.sigma_hat).real
    S_final = solver.theta + sigma_val
    final_points = eta_circle.eval(S_final)
    
    # Calculate distance from origin (should be 1.0 everywhere)
    radii = np.abs(final_points)
    radius_error = np.max(np.abs(radii - 1.0))
    print(f"Max deviation from unit radius: {radius_error}")
    
    assert radius_error < 1e-14, "Final points are not on the unit circle!"


def test_omega_ellipse():
    '''
    test wegmann solver for ellipse target region
    '''
    print("testing wegmann solver for ellipse target region ...")
    N=128
    a = 2.0 # semi-major axis
    b = 1.0 # semi-minor axis
    theta = np.linspace(0,2*np.pi, num=N, endpoint=False)
    eta_vals = a * np.cos(theta) + 1j * b * np.sin(theta)
    eta_coeffs = np.fft.fft(eta_vals)/N
    ellipse_bdary_obj = BoundaryCurve(eta_coeffs)
    solver = WegmannSolver(ellipse_bdary_obj, N)
    print("Running wegmann solver for ellipse target region ...")
    solver.find_conformal_map(max_iter=100, epsilon=1e-10)
    final_pts= solver.get_final_boundary_points() #get final coordinates on boundary after convergence
    # check if pts are on ellipse
    x = final_pts.real
    y = final_pts.imag
    metric = (x/a)**2 + (y/b)**2
    deviation = np.max(np.abs(metric - 1.0))
    print(f"Max deviation from ellipse boundary: {deviation}")
    assert deviation < 1e-5, "FAILURE: Final points are not on the ellipse boundary"
    print("SUCCESS: final points lie on the ellipse boundary")

def test_low_resolution_mesh():
    '''
    check if discretization is coarse, derivatives still make sense or too big errors
    especialy check hilbert transform validity
    '''
    N=8

def test_non_convex_target_region():
    '''
    test if wegmann solver converges for non-convex target region and if not, make it raise an error instead of just crashing
    '''

def test_zero_derivative_boundary():
    '''
    test if wegmann solver handles case where boundary derivative is zero at some point (cusp) (should raise zerodivisionerror)
    '''
#delete this
def construct_conformal_map(self):
    '''
    construct conformal map psi from unit disk to target region from final state (sigma_hat)
    returns fourier coeffs of psi
    '''
    sigma_hat = self.find_conformal_map(max_iter=100, epsilon=1e-6)
    sigma_val = np.fft.ifft(sigma_hat).real
    #shift back the conformal mapping to the actual centroid
    self.sigma_hat[0] += self.shift
    S_final = self.theta + sigma_val
    psi = self.eta.evaluate(S_final)

    return psi
