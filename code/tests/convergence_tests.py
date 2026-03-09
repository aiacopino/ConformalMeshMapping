import numpy as np
import matplotlib.pyplot as plt
from src.wegmann import WegmannSolver
from src.targetregion import BoundaryCurve

'''
in this file we test the algorithm as a whole 
to ensure convergence for the most common target region shapes
'''

def plot_conformal_grid(solver, N, p=0, verfahren=1, relaxation=1, accuracy = np.inf, actual_boundary_pts_original=None):
    '''
    mesh on unit disk
    maybe make the mesh gen a method inside operators or so idk
    
    :param solver: Description
    :param num_circles: number of radial lines (concentric circles in unit disk)
    :param num_lines: number of angular lines (rays from origin in unit disk)must be even ? 
    '''
    if actual_boundary_pts_original is None:
        raise ValueError("actual_boundary_pts_original must be provided for plotting, received None.")
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

    # Plot Actual Boundary if provided

    # Plot Boundary
    status = solver.status if hasattr(solver, 'status') else "UNKNOWN_STATUS"
    bdary = solver.get_final_boundary_points()
    plt.scatter(actual_boundary_pts_original.real, actual_boundary_pts_original.imag, color='green', s=10, zorder=5, label='Original Boundary Grid Pts', alpha=.5)
    plt.scatter(bdary.real, bdary.imag, color='red', s=10, zorder=5, label='Solver Output Grid Pts')
    plt.plot(np.append(bdary.real, bdary.real[0]), np.append(bdary.imag, bdary.imag[0]), 'r--', label='Boundary Traversal Path', alpha=.3) # close the boundary loop
    plt.axis('equal')
    plt.title(f"Conformal Map (N={N}, convexity param p={p}, relaxation={relaxation}) \n Status {status} after {solver.i} iterations of verfahren {verfahren}. \n Accuracy = {accuracy}")
    plt.legend()
    plt.show()

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
    _, solver.status = solver.find_conformal_map(max_iter=100, epsilon=1e-5, relaxation = 1, verfahren=verfahren)

    _check_analyticity(solver, tol = 1e-10)
    plot_conformal_grid(solver, N, p=0, verfahren=verfahren)
    return solver.status

def test_eccentric_circle_convergence(N, relaxation, verfahren, tolerance=1e-10):
    '''
    from Gaier, Anhang 5, first example
    example with known conformal map

    :param N: discretisation points
    '''
    #params as in Gaier and Andersen (not anymore hehe)
    a = 0.2
    b = a/.6

    print(f"START testing wegmann solver convergence for eccentric circle domain target region ... N={N}, a={a}, b={b}")
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = a * np.cos(t) + np.sqrt(b**2 - a**2 * np.sin(t)**2)
    x = r * np.cos(t)
    y = r * np.sin(t)
    eta_vals = x + 1j*y
    eta_coeffs = np.fft.fft(eta_vals)/N
    eccentric_circle_bdary = BoundaryCurve(eta_coeffs)
    #debugging
    print(f"FLAG: center of eta is {eccentric_circle_bdary.coeffs[0]}, should be close to 0 for centered shape. First 5 Fourier coefficients of eccentric circle boundary: {eccentric_circle_bdary.coeffs[:5]}")
    solver = WegmannSolver(eccentric_circle_bdary, N)
    #solver.init_initial_guess_starshaped()
    solver.init_initial_guess_identity()
    _, solver.status = solver.find_conformal_map(max_iter=300, epsilon=tolerance, relaxation=relaxation, verfahren=verfahren)
    # test accuracy
    computed_t = np.unwrap(t + np.fft.ifft(solver.sigma_hat).real)
    computed_r = a * np.cos(computed_t) + np.sqrt(b**2 - a**2 * np.sin(computed_t)**2)
    # conformal map given by Gaier, note his theta is our target angle and his phi is our computed_t
    c = a / ( b**2 - a**2)
    numerator = np.sin(computed_t)
    denom = c*computed_r + np.cos(computed_t)
    known_conformal_phi =  np.unwrap(np.arctan2(numerator, denom)) # arctan2 for correct quadrant handling
    # fixing rotation
    # Wegmann goes t to computed_t, Gaier geos back. if solver is correct the composition should be the identity
    phase_diff = known_conformal_phi - t
    phase_diff -= np.mean(phase_diff) # remove mean to correct for constant rotation ambiguity
    accuracy = float(np.linalg.norm(phase_diff)) / np.sqrt(N) # RMS error of angle difference
    print(f"Distance between computed conformal map and known conformal map on eccentric circle: {accuracy}")
    plot_conformal_grid(solver, N, 0, verfahren, relaxation, accuracy, actual_boundary_pts_original=eta_vals)
    return accuracy, solver.status

def test_inverted_ellipse_convergence(N,p, relaxation, verfahren, initial_guess="identity", tolerance=1e-10):
    '''
    test wegmann solver convergence for inverted ellipse target region
    example 2 of Gaier Anhang 5 and benchmark example in Wegmann's paper
    '''
    print("START testing wegmann solver convergence for inverted ellipse target region ...")
    eta_coeffs = get_inverted_ellipse_coefficients(N, p)
    inverted_ellipse_bdary_obj = BoundaryCurve(eta_coeffs)
    solver = WegmannSolver(inverted_ellipse_bdary_obj, N)
    if initial_guess == "starshaped":
        solver.init_initial_guess_starshaped()
    elif initial_guess == "non_convex":
        solver.init_initial_guess_non_convex()
    else:
        solver.init_initial_guess_identity() # Wegmann's choice for this example, but can diverge for low p
    print(f"Running wegmann solver for inverted ellipse target region, params N={N}, p={p} ...")
    _, solver.status = solver.find_conformal_map(max_iter=10000, epsilon=1e-4, relaxation = relaxation, verfahren=verfahren)
    '''
    if solver.error_history:
        print(f"last 10 error entries after convergence: {solver.error_history[-10:]}")
    else:
        print("Error history is empty, something went wrong with convergence tracking.")
    '''
    # plotting and measuring accuracy of result
    theta_unitc = np.linspace(0, 2*np.pi, N, endpoint=False)
    z_disk = np.exp(1j * theta_unitc)
    #theta_theoretical_conformal = np.arctan2(p * np.sin(theta_unitc), np.cos(theta_unitc))
    #r_theoretical_conformal = np.sqrt(1 - (1 - p**2) * np.cos(theta_theoretical_conformal)**2)
    boundary_pts_theoretical_conformal = (2 * p * z_disk) / ((1 + p) + (1 - p) * z_disk**2)
    computed_mapped_pts = solver.get_final_boundary_points()

    #sigma_val = np.fft.ifft(solver.sigma_hat).real
    #computed_angles = np.mod(theta_unitc + sigma_val, 2*np.pi)
    #anglediff = np.mod(computed_angles - known_conformal_map + np.pi, 2*np.pi) - np.pi
    accuracy = float(np.linalg.norm(computed_mapped_pts - boundary_pts_theoretical_conformal))
    print(f"L^2 Distance between computed conformal map and known conformal map on inverted ellipse, p = {p}: {accuracy}")
    plot_conformal_grid(solver, N, p, verfahren, relaxation, accuracy, boundary_pts_theoretical_conformal)
    return accuracy, solver.status

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

def test_kite_convergence(N, relaxation, verfahren, tolerance):
    '''
    test convergence for kite domain target region
    k(s) = (cos(s) + 0.65* cos(2s) - 0.65, 1.5 * sin(s)) for s in [0,2*pi]

    '''
    print(f"START testing wegmann solver convergence for kite domain target region, N={N} ...")
    s = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(s) + 0.65 * np.cos(2*s) - 0.65
    y = 1.5 * np.sin(s)
    eta_vals = x + 1j*y
    eta_coeffs = np.fft.fft(eta_vals)/N
    print(f"FLAG: Fourier coefficients of kite boundary: {eta_coeffs}")
    kite_bdary_obj = BoundaryCurve(eta_coeffs)
    solver = WegmannSolver(kite_bdary_obj, N)
    solver.init_initial_guess_identity()
    #solver.init_initial_guess_starshaped()
    #solver.init_initial_guess_non_convex()
    _, solver.status = solver.find_conformal_map(max_iter=10000, epsilon=tolerance, relaxation = relaxation, verfahren=verfahren)

    plot_conformal_grid(solver, N, p=0, verfahren=verfahren, relaxation=relaxation, accuracy=None, actual_boundary_pts_original=eta_vals)
    _check_analyticity(solver, tol = 1e-5)
    return solver.status

def test_square_convergence(tolerance, N, relaxation = 0.1, verfahren = 1, cutoff = 1/3):
    #smoothed square with side length 2 and centered at the origin, like in Gaier Anhang 5 Bsp 4
    #step 1: create fourier coeffs for smoothed square boundary
    t = np.linspace(0,2*np.pi, N, endpoint=False)
    r = 1 / np.maximum(np.abs(np.cos(t)), np.abs(np.sin(t))) # radius for smoothed square boundary, r = 1/max(|cos(t)|, |sin(t)|)
    eta_vals = r * np.exp(1j*t)
    coeffs = np.fft.fft(eta_vals) / N
    #truncation: zero out high frequencies (middle of the array)
    cutoff = int(cutoff * N) # cutoff is half the ratio of coeffs to keep, so cutoff=1/3 means keep 2/3 of coeffs
    coeffs[cutoff:N-cutoff] = 0
    print(f"START testing wegmann solver convergence for smoothed square target region ... N={N}, cutoff={cutoff}")
    square_bdary_obj = BoundaryCurve(coeffs)
    solver = WegmannSolver(square_bdary_obj, N)
    solver.init_initial_guess_starshaped()
    print("running Wegmann solver for smoothed square target region ...")
    _, solver.status = solver.find_conformal_map(max_iter=100000, epsilon=tolerance, relaxation = relaxation, verfahren=verfahren)
    
    #check accuracy of result by comparing to known conformal map for non-smoothed square (Schwarz-Christoffel map)
    computed_mapped_pts = solver.get_final_boundary_points()
    # note that max (L^infty) of all points on this square is one, so we can just check component-wise if the computed points are far from the distance one lines
    x_vals = computed_mapped_pts.real
    y_vals = computed_mapped_pts.imag
    maxnorm_computed_radii = np.maximum(np.abs(x_vals), np.abs(y_vals))
    accuracy = float(np.linalg.norm(maxnorm_computed_radii - 1)) / np.sqrt(N) #RMS error
    print(f"L^2 Geometric Error (Distance to PERFECT square bounds): {accuracy} \n Note accuracy is not expected to be very small due to truncation error. Current truncation cutoff: {cutoff} out of {N} coefficients.")
    plot_conformal_grid(solver, N, 0, verfahren, relaxation, accuracy, eta_vals)
    return accuracy, solver.status

def test_flower_convergence(N, p, relaxation, verfahren, initial_guess, tolerance, petals = 5):
    '''
    Docstring for test_starfish_convergence
    
    :param N: Description
    :param p: Description
    '''
    print(f"START testing wegmann solver convergence for starfish domain target region ... N={N}, convexity param p={p}")
    # cartesian, fixed radius
    R = 3
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = R + p * np.cos(5 * t) # p controls convexity: p close to 0 is more convex
    x = np.cos(t)*(R + p * np.cos(petals * t))
    y = np.sin(t)*(R + p * np.cos(petals * t)) 
    eta_vals = x + 1j*y
    eta_coeffs = np.fft.fft(eta_vals)/N
    print(f"FLAG: Fourier coefficients of flower/star boundary: {eta_coeffs}")
    star_bdary_obj = BoundaryCurve(eta_coeffs)
    solver = WegmannSolver(star_bdary_obj, N)
    if initial_guess == "starshaped":
        solver.init_initial_guess_starshaped()
    else:
        solver.init_initial_guess_identity() # faster to compute, but can diverge for low p
    #solver.find_conformal_map(max_iter=1000, epsilon=5e-2, relaxation = relaxation, verfahren=verfahren)
    _, solver.status = solver.find_conformal_map(max_iter=10000, epsilon=tolerance, relaxation = relaxation, verfahren=verfahren)

    # plotting and measuring accuracy of result
    computed_mapped_pts = solver.get_final_boundary_points()
    computed_radii = np.abs(computed_mapped_pts)
    accuracy = float(np.linalg.norm(computed_radii - r)) / np.sqrt(N) #RMS error
    print(f"L^2 Distance between computed conformal map and known conformal map on flower shape, p = {p}: {accuracy}")
    plot_conformal_grid(solver, N, p, verfahren, relaxation, accuracy, eta_vals)
    return accuracy, solver.status

def _check_analyticity(solver, tol):
    '''
    check the computed psi is actually analytic
    cauchy-riemann equations: u_x = v_y and u_y = -v_x where psi = u + iv
    equivalent to checking f taylor series expansion is well-def, i.e. all negative foourier coeffs are zero

    :param tol: computational error tolerance
    '''
    if solver.status != "CONVERGED":
        print(f"Warning: solver status is {solver.status}. sKIPPING analyticity check.")
        return
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

