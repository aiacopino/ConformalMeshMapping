import numpy as np
import matplotlib.pyplot as plt
from src.wegmann import WegmannSolver
from src.targetregion import BoundaryCurve

'''
in this file we test the algorithm as a whole 
to ensure convergence for the most common target region shapes
'''

def get_inverted_ellipse_coefficients(N,p):
    '''
    helper function to get fourier coefficients of inverted ellipse boundary parametrisation
    benchmark example from wegmann's paper
    parameter p in (0,1): closer to 1 equals ellipse, closer to 0 is more non-convex with sharper cusp at x=0, y=+-1
    '''
    #N = 128
    #p = 0.4 # p in (0,1)
    s = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = np.sqrt(1 - (1 - p**2) * np.cos(s)**2)
    #cartesian frmo aobve angle s and radius r
    x = r * np.cos(s)
    y = r * np.sin(s)
    eta_vals = x + 1j*y
    eta_coeffs = np.fft.fft(eta_vals)/N
    return eta_coeffs
    ''' (no more fun)
    #plotting for fun
    plt.figure(figsize = (7,7))
    plt.plot(x,y,label=f'Inverted Ellipse with p={p}')
    plt.axis('equal') #distortion prevention
    plt.legend()
    plt.title(f"Wegmann's Inverted Ellipse Example (1978), N={N}, p={p}")
    plt.show()
    '''

def check_analyticity(solver, tol):
    '''
    check the computed psi is actually analytic
    cauchy-riemann equations: u_x = v_y and u_y = -v_x where psi = u + iv
    equivalent to checking f taylor series expansion is well-def, i.e. all negative foourier coeffs are zero

    :param tol: computational error tolerance
    '''
    print("checking analyticity of computed conformal map ...")
    bdary_pts = solver.get_final_boundary_points()
    coeffs = np.fft.fft(bdary_pts)/solver.N
    negative_coeffs = coeffs[solver.N//2+1:]
    ncoeffs_magnitude = np.linalg.norm(negative_coeffs)
    print(f"Norm of negative Fourier coefficients (should be close to 0 for analyticity): {ncoeffs_magnitude}")
    if ncoeffs_magnitude > tol:
        print(f"Error: Negative Fourier coefficients have norm {ncoeffs_magnitude} > {tol}. computed map not analytic")
    else:
        print("Analyticity check passed: negative Fourier coefficients negligeable.")

def plot_conformal_grid(solver, N, p=0, verfahren=1, relaxation=1):
    '''
    mesh on unit disk
    maybe make the mesh gen a method inside operators or so idk
    
    :param solver: Description
    :param num_circles: number of radial lines (concentric circles in unit disk)
    :param num_lines: number of angular lines (rays from origin in unit disk)must be even ? 
    '''
    num_circles = 10
    num_lines = N
    print("plotting conformal grid ...")

    r = np.linspace(0, 1, num_circles) # radial points
    theta = np.linspace(0, 2*np.pi, num_lines, endpoint=True) # angular points
    
    # Plot Circles
    plt.figure(figsize=(8, 8))
    for radius in r:
        z_circle = radius * np.exp(1j * theta)
        w_circle = solver.calculate_interior_mesh(z_circle)
        plt.plot(w_circle.real, w_circle.imag, 'b-', alpha=0.5, linewidth=0.8)
        
    # Plot Rays
    r_ray = np.linspace(0, 1.0, 100)
    for angle in theta:
        z_ray = r_ray * np.exp(1j * angle)
        w_ray = solver.calculate_interior_mesh(z_ray)
        plt.plot(w_ray.real, w_ray.imag, 'k-', alpha=0.4, linewidth=0.8)

    # Plot Boundary
    bdary = solver.get_final_boundary_points()
    plt.scatter(bdary.real, bdary.imag, color='red', s=10, zorder=5, label='Solver Output Grid Pts')
    plt.plot(np.append(bdary.real, bdary.real[0]), np.append(bdary.imag, bdary.imag[0]), 'r-', label='Boundary') # close the boundary loop
    plt.axis('equal')
    plt.title(f"Conformal Map (N={N}, convexity param p={p}, relaxation={relaxation}) \n converged in {solver.i} iterations of verfahren {verfahren}")
    plt.legend()
    plt.show()

def test_inverted_ellipse_convergence(N,p, relaxation, verfahren):
    '''
    test wegmann solver convergence for inverted ellipse target region
    example 2 of Gaier Anhang 5 and benchmark example in Wegmann's paper
    '''
    print("START testing wegmann solver convergence for inverted ellipse target region ...")
    eta_coeffs = get_inverted_ellipse_coefficients(N, p)
    inverted_ellipse_bdary_obj = BoundaryCurve(eta_coeffs)
    solver = WegmannSolver(inverted_ellipse_bdary_obj, N)
    solver.init_initial_guess_starshaped()
    print(f"Running wegmann solver for inverted ellipse target region, params N={N}, p={p} ...")
    solver.find_conformal_map(max_iter=10000, epsilon=1e-4, relaxation = relaxation, verfahren=verfahren)

    if solver.error_history:
        print(f"last 10 error entries after convergence: {solver.error_history[-10:]}")
    else:
        print("Error history is empty, something went wrong with convergence tracking.")
    
    # plotting and measuring accuracy of result
    theta_unitc = np.linspace(0, 2*np.pi, N, endpoint=False)
    z_disk = np.exp(1j * theta_unitc)
    #theta_theoretical_conformal = np.arctan2(p * np.sin(theta_unitc), np.cos(theta_unitc))
    #r_theoretical_conformal = np.sqrt(1 - (1 - p**2) * np.cos(theta_theoretical_conformal)**2)
    boundary_pts_theoretical_conformal = (2 * p * z_disk) / ((1 + p) + (1 - p) * z_disk**2)
    computed_mapped_pts = solver.final_bdary_pts

    #sigma_val = np.fft.ifft(solver.sigma_hat).real
    #computed_angles = np.mod(theta_unitc + sigma_val, 2*np.pi)
    #anglediff = np.mod(computed_angles - known_conformal_map + np.pi, 2*np.pi) - np.pi
    distance_to_known_map = float(np.linalg.norm(computed_mapped_pts - boundary_pts_theoretical_conformal))
    print(f"L^2 Distance between computed conformal map and known conformal map on inverted ellipse, p = {p}: {distance_to_known_map}")
    plot_conformal_grid(solver, N, p, verfahren, relaxation)

def test_unit_disk_convergence(N, verfahren):
    '''
    test convergence for unit disk target region, should be the identity map
    '''
    print("START testing wegmann solver convergence for unit disk target region ...")
    eta_coeffs = np.zeros(N, dtype=complex)
    eta_coeffs[1] = 1.0 # unit circle
    unit_disk_bdary_obj = BoundaryCurve(eta_coeffs)
    solver = WegmannSolver(unit_disk_bdary_obj, N)

    solver.init_initial_guess_identity()
    solver.find_conformal_map(max_iter=100, epsilon=1e-5, relaxation = 1, verfahren=verfahren)

    check_analyticity(solver, tol = 1e-10)
    plot_conformal_grid(solver, N, verfahren)

def test_kite_convergence(N, relaxation, verfahren):
    '''
    test convergence for kite domain target region
    k(s) = (cos(s) + 0.65* cos(2s) - 0.65, 1.5 * sin(s)) for s in [0,2*pi]

    '''
    print("START testing wegmann solver convergence for kite domain target region ...")
    s = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(s) + 0.65 * np.cos(2*s) - 0.65
    y = 1.5 * np.sin(s)
    eta_vals = x + 1j*y
    eta_coeffs = np.fft.fft(eta_vals)/N
    print(f"FLAG: Fourier coefficients of kite boundary: {eta_coeffs}")
    kite_bdary_obj = BoundaryCurve(eta_coeffs)
    solver = WegmannSolver(kite_bdary_obj, N)
    #solver.init_initial_guess_starshaped()
    solver.init_initial_guess_non_convex()
    solver.find_conformal_map(max_iter=1000, epsilon=1e-5, relaxation = relaxation, verfahren=verfahren)

    plot_conformal_grid(solver, N, verfahren, relaxation)
    check_analyticity(solver, tol = 1e-5)

def test_starfish_convergence(N, p, relaxation, verfahren):
    '''
    Docstring for test_starfish_convergence
    
    :param N: Description
    :param p: Description
    '''
    print(f"START testing wegmann solver convergence for starfish domain target region ... N={N}, convexity param p={p}")
    # cartesian, fixed radius
    R = 3
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = R + p * np.cos(5 * t) # starfish shape, p controls convexity: p close to 0 is more convex
    x = np.cos(t)*(R + p * np.cos(5 * t))
    y = np.sin(t)*(R + p * np.cos(5 * t)) 
    eta_vals = x + 1j*y
    eta_coeffs = np.fft.fft(eta_vals)/N
    print(f"FLAG: Fourier coefficients of starfish boundary: {eta_coeffs}")
    starfish_bdary_obj = BoundaryCurve(eta_coeffs)
    solver = WegmannSolver(starfish_bdary_obj, N)
    solver.init_initial_guess_starshaped()
    #solver.find_conformal_map(max_iter=1000, epsilon=5e-2, relaxation = relaxation, verfahren=verfahren)
    solver.find_conformal_map(max_iter=1000, epsilon=1e-4, relaxation = relaxation, verfahren=verfahren)
    plot_conformal_grid(solver, N, p, verfahren, relaxation)
    check_analyticity(solver, tol = 1e-5)

def test_eccentric_circle_convergence(N, relaxation, verfahren):
    '''
    from Gaier, Anhang 5, first example
    example with known conformal map

    :param N: discretisation points
    '''
    #params as in Gaier and Andersen (not anymore hehe)
    a = 0.55
    b = 0.6

    print(f"START testing wegmann solver convergence for eccentric circle domain target region ... N={N}, a={a}, b={b}")
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = a * np.cos(t) + np.sqrt(b**2 - a**2 * np.sin(t)**2) # eccentric circle with foci at x=+-a, radius b
    # conformal map given by Gaier, note his theta is out target angle and his phi is our t
    numerator = b * np.sin(t)
    denom = b * np.cos(t) - a
    known_conformal_phi =  np.arctan2(numerator, denom) # arctan2 for correct quadrant handling
    x = r * np.cos(t)
    y = r * np.sin(t)
    eta_vals = x + 1j*y
    eta_coeffs = np.fft.fft(eta_vals)/N
    eccentric_circle_bdary = BoundaryCurve(eta_coeffs)
    solver = WegmannSolver(eccentric_circle_bdary, N)
    solver.init_initial_guess_starshaped()
    solver.find_conformal_map(max_iter=1000, epsilon=1e-5, relaxation=relaxation, verfahren=verfahren)
    computed_phi = solver.get_final_boundary_points() 
    plot_conformal_grid(solver, N, 0, verfahren, relaxation)
    distance_to_known_map = float(np.linalg.norm(computed_phi - known_conformal_phi))
    print(f"Distance between computed conformal map and known conformal map on eccentric circle: {distance_to_known_map}")

if __name__ == "__main__":
    # adapt params here for different target regions and convergence tests
    N = 512
    p = 0.3
    relaxation = 0.01

    # verfahren 1 or 2 from wegmann's paper, one is more exact and two converges faster.
    # note any value other than 1 will run verfahren 2.
    # (note verfahren translates to "method" but we chose the german word because method in non-py programming is a fct; to avoid confusion)
    verfahren = 2

    test_inverted_ellipse_convergence(N, p, relaxation, verfahren) #converges for relaxation >0.8
    #test_kite_convergence(N, relaxation, verfahren)
    #test_unit_disk_convergence(N, verfahren)
    #test_starfish_convergence(N, p, relaxation, verfahren)
    #test_eccentric_circle_convergence(N, relaxation, verfahren)