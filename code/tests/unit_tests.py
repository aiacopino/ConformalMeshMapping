import numpy as np
import matplotlib.pyplot as plt
import src.operators as ops
from src.wegmann import WegmannSolver
from src.targetregion import BoundaryCurve
'''
in this file we test every component of the algorithm separately
'''
def test_hilbert_transform():
    '''
    H(cos) = sin, H(sin) = -cos
    '''
    print("testing hilbert transform ...")
    N=128
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    cos_vals = np.cos(t)
    sin_vals = np.sin(t)
    H_cos = ops.hilbert_transform(cos_vals)
    H_sin = ops.hilbert_transform(sin_vals)
    assert np.allclose(H_cos, sin_vals, atol=1e-10)
    assert np.allclose(H_sin, -cos_vals, atol=1e-10)
    print("Hilbert transform tests passed.")

def test_initial_guess():
    '''
    check if boundary corerspondence function (including initial guess) maps from unit circle to [0,2pi]
    '''
    print("testing initial guess for boundary correspondence function ...")
    N=2048
    dummy_eta = np.zeros(N, dtype=complex)
    solver = WegmannSolver(dummy_eta, N)
    solver.init_initial_guess()
    assert np.equal(solver.sigma_hat, np.zeros(N, dtype=complex)).all(), "Error: Initial guess for sigma_hat is not zero array."
    print("Initial guess for sigma_hat is correct (zero array).")

    # test if S is the identity map
    sigma_val = np.fft.ifft(solver.sigma_hat).real
    S_k = solver.theta + sigma_val
    print(f"S_k min value: {np.min(S_k)} (expected 0), max value: {np.max(S_k)} (expected close to 2pi (endpt=False in linspace))")

def test_boundary_geometry_evaluation_easy():
    '''
    test if the boundary geomtry evaluation is actually on the boundary (f_k(S_k) is on the boundary but not yet at the correct place)
    '''
    print("testing boundary geometry evaluation ...")
    N = 128
    eta_dummy = np.zeros(N, dtype=complex)
    eta_dummy[1] = 1.0 # unit circle
    solver = WegmannSolver(eta_dummy, N)
    # check if S_k = theta + sigma_k where sigma_k is 2pi-periodic (check if sigma_k(0)=sigma_k(2pi))
    sigma_spatial = np.full(N, 0.1)
    solver.sigma_hat = np.fft.fft(sigma_spatial)

    S_expected = solver.theta + sigma_spatial
    f_expected = np.exp(1j * S_expected)
    g_expected = 1j * np.exp(1j * S_expected)

    f_k, g_k = solver.evaluate_boundary_geometry()

    assert np.allclose(f_k, f_expected, atol=1e-10), "Error: Evaluated boundary points f_k do not match expected values."
    assert np.allclose(g_k, g_expected, atol=1e-10), "Error: Evaluated boundary tangents g_k do not match expected values."
    print("Boundary geometry evaluation tests passed.")

def test_boundary_geometry_evaluation_ellipse_vary():
    '''
    test on an ellipse wit semi axes 2 and 1
    reparametrisation sigma = 0.2sin(theta) lets see what happens
    '''
    print("testing boundary geometry evaluation on ellipse with varying reparametrisation ...")
    N = 32
    a = 2.0
    b = 1.0
    theta = np.linspace(0, 2*np.pi, num=N, endpoint=False)
    # eta(t) = 1.5 e^{it} + 0.5 e^{-it}
    eta_vals = a * np.cos(theta) + 1j * b * np.sin(theta)
    eta_coeffs = np.fft.fft(eta_vals)/N 
    print(f"FLAG: Fourier coefficients of ellipse boundary: {eta_coeffs}")
    # isclose rather than == because of float eval FFT inexactness
    assert (np.isclose(eta_coeffs [1], 1.5, atol=1e-10) and np.isclose(eta_coeffs[N-1], 0.5, atol=1e-10)), "Error: Fourier coefficients of ellipse boundary do not match expected values."
    solver = WegmannSolver(eta_coeffs, N)
    sigma_spatial = 0.2 * np.sin(solver.theta)
    solver.sigma_hat = np.fft.fft(sigma_spatial)
    S_k = solver.theta + sigma_spatial
    f_expected = a * np.cos(S_k) + 1j * b * np.sin(S_k)
    g_expected = -a * np.sin(S_k) + 1j * b * np.cos(S_k)
    f_k, g_k = solver.evaluate_boundary_geometry()
    assert np.allclose(f_k, f_expected, atol=1e-10), "Error: Evaluated boundary points f_k do not match expected values."
    assert np.allclose(g_k, g_expected, atol=1e-10), "Error: Evaluated boundary tangents g_k do not match expected values."
    print("Boundary geometry evaluation on ellipse with varying reparametrisation tests passed.")

def test_RH_solver():
    '''
    check if solution is given analytic fct psi
    '''
    N=32
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.exp(1j*t)
    dummy_eta = np.zeros(N, dtype=complex) # not used in this test but needed to init solver instance
    solver = WegmannSolver(dummy_eta, N)
    psi_true = z**2
    A_dummy = 1.0 + .5*z # has winding number 0
    RHS_dummy = np.real(psi_true/A_dummy)
    print(f"testing RH solver with A={A_dummy}, RHS={RHS_dummy}")
    psi_computed = solver.solve_RH_problem(A_dummy, RHS_dummy)
    assert np.allclose(psi_computed, psi_true, atol=1e-10)
    print(f"Riemann-Hilbert solver test passed.")

def test_RH_solver_trivial():
    '''
    test for a constant A and RHS
    '''
    N=32
    z = np.exp(1j* np.linspace(0, 2*np.pi, N, endpoint=False))
    dummy_eta = np.zeros(N, dtype=complex) # not used in this test but needed to init solver instance
    solver = WegmannSolver(dummy_eta, N)
    A_dummy = np.ones(N) # constant A with winding number 0
    RHS_dummy = np.ones(N) # constant RHS
    print(f"testing RH solver with constant A={A_dummy}, constant RHS={RHS_dummy}")
    psi_computed = solver.solve_RH_problem(A_dummy, RHS_dummy)
    LHS = (psi_computed / A_dummy).real
    assert np.allclose(LHS, RHS_dummy, atol=1e-10)
    print(f"Riemann-Hilbert solver trivial test passed.")

def test_RH_solver_exponential_magnitude():
    '''
    test for exponentially varying magnitude A
    '''
    N=32
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.exp(1j* t)
    dummy_eta = np.zeros(N, dtype=complex) # not used in this test but needed to init solver instance
    solver = WegmannSolver(dummy_eta, N)
    A_dummy = np.exp(np.cos(t)) # varying magnitude A with winding number 0
    psi_true = np.exp(z)
    RHS_dummy = (psi_true/A_dummy).real
    print(f"testing RH solver with varying magnitude A={A_dummy}, RHS={RHS_dummy}")
    psi_computed = solver.solve_RH_problem(A_dummy, RHS_dummy)
    LHS = (psi_computed / A_dummy).real
    assert np.allclose(LHS, RHS_dummy, atol=1e-10)
    print(f"Riemann-Hilbert solver varying magnitude test passed.")

def test_RH_solver_oscillatory_phase():
    '''
    test for oscillatory phase A
    '''
    N= 2048 # 256 # 32
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.exp(1j* t)
    dummy_eta = np.zeros(N, dtype=complex) # not used in this test but needed to init solver instance
    solver = WegmannSolver(dummy_eta, N)
    A_dummy = np.exp(1j * 5 * np.sin(t)) # oscillatory phase A from -5 to 5, winding number 0
    psi_true = 1+z**12409 # z**3
    RHS_dummy = (psi_true/A_dummy).real
    # print(f"testing RH solver with oscillatory phase A={A_dummy}, RHS={RHS_dummy}") # too much output, comment out for now
    psi_computed = solver.solve_RH_problem(A_dummy, RHS_dummy)
    LHS = (psi_computed / A_dummy).real
    assert np.allclose(LHS, RHS_dummy, atol=1e-10)
    print(f"Riemann-Hilbert solver oscillatory phase test passed.")

def test_RH_solver_meromorphic():
    '''
    test for meromorphic psi with a pole just outside the unit disk
    '''
    N= 2048 # 256 # 32
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.exp(1j* t)
    dummy_eta = np.zeros(N, dtype=complex) # not used in this test but needed to init solver instance
    solver = WegmannSolver(dummy_eta, N)
    A_dummy = (1+.5*z.real)*np.exp(1j*np.sin(np.angle(z))) 
    psi_true = 1/(z-1.1) # meromorphic psi with a pole outside the unit disk
    RHS_dummy = (psi_true/A_dummy).real
    psi_computed = solver.solve_RH_problem(A_dummy, RHS_dummy)
    LHS = (psi_computed / A_dummy).real
    error = np.max(np.abs(psi_computed - psi_true))
    print(f"meromorphic error, N={N}: {error}")
    # assert np.allclose(psi_true, psi_computed, atol=1e-5) # not working
    # print(f"Riemann-Hilbert solver meromorphic test passed.")

if __name__ == "__main__":
    test_initial_guess()
    #test_omega_unit_circle()
    #test_omega_ellipse()
    #test_hilbert_transform()
    #test_RH_solver_trivial()
    #test_RH_solver_exponential_magnitude()
    #test_RH_solver_oscillatory_phase()
    #test_RH_solver_meromorphic()
    #test_boundary_geometry_evaluation_ellipse_vary()