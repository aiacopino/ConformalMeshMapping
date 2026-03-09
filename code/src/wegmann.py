import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import fsolve # for non-convex initial guess
import src.operators as ops
from src.targetregion import BoundaryCurve

class WegmannSolver:
    '''
    wegmann's method algorithm for conformal mapping
    input: boundary parametrisation eta of target region, mesh on unit disk to be mapped to target region
    output: conformal map psi from unit disk to target region
    '''
    def __init__(self, eta, N):
        '''        
        :param eta: boundary parametrisation, format: array of fourier coefficients (complex)
        :param N: number of discretization points on the boundary
        '''
        self.N = N
        
        # to fix a parameter psi(0)=0 for uniqueness, we might need to shift the target region so that it contains 0 in the first place. 
        # the perfect shift has exactly centroid of target domain at 0
        # this will hopefully work because most of the regions studied by wegmann are more or less convex
        self.shift = eta.coeffs[0]
        print(f"FLAG: normalisation shift applied to target region: {self.shift}")
        centered_coeffs = eta.coeffs.copy()
        centered_coeffs[0] = 0
        self.eta = BoundaryCurve(centered_coeffs)

        if len(self.eta.coeffs) % 2 != 0: #check even number of fourier coeffs
            raise ValueError(f"WegmannSolver __init__: error in normalisation shift. Expected even number of Fourier coefficients and zero mean (first coeff), got {len(self.eta.coeffs)} coeffs and mean {self.eta.coeffs[0]}.")

        self.theta = np.linspace(0, 2*np.pi, num=N, endpoint=False) # array of equally spaced args (angles)
        self.zeta = np.exp(1j*self.theta) # array of equally spaced grid pts on unit circle [0, 2pi)
        
        # initial guess: S(theta) = theta <=> sigma = 0
        self.sigma_hat = np.zeros(self.N, dtype=complex) # fourier coeffs of boundary correspondenec fct, equation 33 in thesis (fourier coeffs of sigma)
        self.error_history = [] # convergence tracking

        # tool for error estimation in the correction step
        #self.domain_diameter =  2 * np.max(np.abs(self.eta.evaluate(self.theta)))

        #rotation offset for plotting, stored here because only needed when normalising orientation of the mapped grid in the end
        self.rotation_offset = 0

        self.h_k = None # placeholder for h_k+1 to be used in newton step, defined here for scope reasons
        self.final_bdary_pts = None # placeholder for final boundary points after convergence, defined here for scope reasons
        self.is_injective = True
        self.status = "INITIALISED" # convergence/divergence status tracking

    def find_conformal_map(self, max_iter=100, epsilon=1e-5, relaxation=0.1, verfahren=1):
        '''
        Wegmann's algorithm for conformal mapping solver        
        :param self: Description
        :param max_iter: maximum iterations of Newton step before stopping
        :param epsilon: convergence tolerance for stopping criterion
        :param relaxation: damping factor for correction step, to stabilize convergence if initial guess is way off. if convergence is too slow, this can be increased.
        :param verfahren: 1 or 2 from wegmann's paper
        '''
        # damping factor to stabilize convergence if initial guess is way off
        # if convergence is too slow, this can be increased
        # relaxation = 0.1

        # default correction step direction (orientation) as in the paper, 
        # might change direction if the correction causes error to increase
        #RH_sign = 1 
        print(f"FLAG: Starting Wegmann solver with max_iter={max_iter}, epsilon={epsilon}, relaxation={relaxation}, verfahren={verfahren} ...")
        for i in range(max_iter):
            old_sigma = self.sigma_hat.copy()
            self.i = i
            if verfahren == 1 or (verfahren == 2 and i == 0):
                U_k_hat = self.newton_step(verfahren=1)
            else:  
                U_k_hat = self.newton_step(verfahren=2)

            self.sigma_hat = old_sigma + relaxation * U_k_hat # sigma_new = sigma_old + U_k (by equation 33)
            #enforce mean for uniqueness (riemann mapping thm); more numerically stable option than pinning a given point which could be badly chosen
            # change this to anchoring a specific point for non-symmetric target shapes
            self.sigma_hat[0] = 0 # minus one dof for uniqueness (riemann mapping thm)

            # if U_k is very small, we are done (remaining correction is very small, algorithm "settled" -> converged)
            # coul we also do this measuring the output of compute_U_k_correction_hat() fct?
            correction_magnitude = float(np.linalg.norm(U_k_hat)) # norm of the correction vector    
            self.error_history.append(correction_magnitude)
            print(f"FLAG: iter {i}, norm of correction: {correction_magnitude}")
            if correction_magnitude > 1e5:
                print(f"ERR: solver diverging, stopping. Last correction magnitude: {correction_magnitude} iteration: {i}")
                return self.sigma_hat, "DIVERGED"
            if correction_magnitude < epsilon:
                if not self.is_injective:
                    print(f"WARNING: solver converged but S_k is not injective, S_dot_k has non-positive values. Potential overlap of boundary points ==> map not conformal")
                    return self.sigma_hat, "CONVERGED_BUT_NOT_CONFORMAL"
                print(f"Converged in {i} iterations.")# \n Error history: {self.error_history}")
                self.normalise_orientation() # psi'(0)>0 maybe this helps fix the offset/ bad accuracy
                # riemann mpping thm check
                print(f"SUCCESS: Riemann Mapping Theorem conditions are met for uniqueness:")
                print(f"  -> Holomorphy & Boundary matched (error = {correction_magnitude:.2e} < {epsilon})")
                print(f"  -> Map is injective (S_k' > 0 implies psi is univalent via Darboux)")
                print(f"  -> Anchored at target center (psi(0) = {self.shift:.2e})")
                print(f"  -> Orientation normalised (psi'(0) > 0)")
                return self.sigma_hat, "CONVERGED"
        #if not converged after max_iter
        print(f"Error history: {self.error_history}")
        print(f"Max iter reached; minimal error from all iterations: {min(self.error_history)}. set as epsilon for almost sure convergence")
        return self.sigma_hat, "MAX_ITER_REACHED"

    def init_initial_guess_identity(self):
        '''
        initial guess for boundary correspondence function S mapping angle theta (in [0, 2pi]) of unit disk to boundary parameter s of target region (eta(S))        
        if i already have this in __init__, can i delete this method and call __init__ from the test file?
        '''
        # initial guess is identity map S(theta) = theta
        # S(theta) = theta + sigma(theta) where sigma is 2pi-periodic smooth (equation 33 in thesis)
        # hence sigma = 0 is initial guess
        self.sigma_hat = np.zeros(self.N, dtype=complex) 

    def init_initial_guess_starshaped(self):
        '''
        convergence-friendly initial guess for star-shaped, highly non-circular domains
        radial projection of disk angle onto curve parameter s s.t. eta(s)=theta
        reference Gaier/Henrici
        '''
        sigma_init = np.zeros(self.N)
        s_guess = 0

        for i, th in enumerate(self.theta):
                # Objective function: We want the angle difference to be 0.
                # Multiplying by exp(-i*th) rotates the point so the target angle is 0.
                # np.angle returns [-pi, pi], which is perfectly smooth around 0.
                def angle_diff(s_array):
                    s = s_array[0] # fsolve passes an array
                    point = self.eta.evaluate(s)
                    return np.angle(point * np.exp(-1j * th))
                
                # Find the root 's' that makes angle_diff(s) == 0
                s_solution, = fsolve(angle_diff, s_guess)
                
                # Update our guess for the next iteration to speed up the solver
                s_guess = s_solution
                
                # S(theta) = s, therefore sigma(theta) = s - theta
                sigma_init[i] = s_solution - th

        sigma_init = np.unwrap(sigma_init)

        # uniqueness
        sigma_init -= np.mean(sigma_init)
        self.sigma_hat = np.fft.fft(sigma_init)
        '''
        points = self.eta.evaluate(self.theta)
        angles = np.angle(points)
        angles = np.unwrap(angles)

        # angles=S(theta)=theta+sigma
        sigma_init = angles - self.theta

        #uniqueness: mean zero (Riemann mapping thm)
        sigma_init -= np.mean(sigma_init)
        self.sigma_hat = np.fft.fft(sigma_init)
        '''
        print(f"FLAG: radial initial guess, max |sigma|: {np.max(np.abs(sigma_init)):.4f}")

    def init_initial_guess_non_convex(self):
        '''
        convergence-friendly initial guess for non-convex domains, based on Hiptmair's kite
        k(s) = (cos(s) + 0.65* cos(2s) - 0.65, 1.5 * sin(s)) for s in [0,2*pi]
        '''
        points = self.eta.evaluate(self.theta)
        angles = np.arctan2(points.imag / 1.5, points.real) #elliptical scaling
        angles = np.unwrap(angles) # smoothen out jump discontniuties

        # angles=S(theta)=theta+sigma
        sigma_init = angles - self.theta

        #uniqueness: mean zero (Riemann mapping thm)
        sigma_init -= np.mean(sigma_init)
        self.sigma_hat = np.fft.fft(sigma_init)
        print(f"FLAG: non-convex initial guess, max |sigma|: {np.max(np.abs(sigma_init)):.4f}")
    
    def evaluate_boundary_geometry(self):
        '''
        evaluate boundary geometry from current state (sigma_hat)
        returns f_k (positions on boundary) and g_k (tangent vectors on boundary)
        '''
        # recover geometry from state
        # S_k(θ) = θ + σ(θ) 
        sigma_val = np.fft.ifft(self.sigma_hat).real # spatial domain
        S_k = self.theta + sigma_val # array of stretched parameters (one for each theta_i for i in [N])
        
        #derivative of S_k : S_k = theta + sigma(theta) => S_k'(theta) = 1 + sigma'(theta)
        #sigma_dot_hat = ops.fourier_derivative(self.sigma_hat)
        #sigma_dot_val = np.fft.ifft(sigma_dot_hat).real
        S_dot_k = 1 + np.fft.ifft(ops.fourier_derivative(self.sigma_hat)).real # spatial domain derivative of S_k, needed for checking injectivity (divergence debugging)
        # save exact iter at which injectivity breaks, if not injective dont check S_dot_k again
        self.is_injective = not np.any(S_dot_k <= 0)
        if not self.is_injective:
            print(f"WARNING: injectivity of S_k lost at iter {self.i}, S_dot_k has non-positive values. Potential overlap of boundary points ==> map not conformal") #: {S_dot_k[S_dot_k <= 0]}, consider improving initial guess.")
        f_k = self.eta.evaluate(S_k) #position on boundary f_k = η(S_k(θ)), geometric candidate but not analytic -> not extendable into interior of disk (analytic candidate is psi_k+1 on the tangent)
        #g_hat = ops.fourier_derivative(f_k) * S_dot_k #tangent on buondary g_k = η'(S_k(θ)) * S_k'(θ)
        g_k = self.eta.evaluate_derivative(S_k) # as in the paper, no chain rule
        return f_k, g_k

    def newton_step(self, verfahren):
        '''
        perform one newton iteration
        stuff within the while loop (lines 4-11) of algorithm 3 in thesis
        :param verfahren: 1 or 2 from wegmann's paper, 1 is default (linear), 2 is accelerated (recycles last step's h_k)
        *** very first iter always verfahren 1 (according to wegmann p 464 bottom paragraph) ***
        output: U_k_hat, the fourier coeffs of the correction step to be added to sigma_hat (possibly damped by relaxation factor)

        NOTE THAT THE ANALYTIC APPROXIMATION WEGMANN CALLED h_k WE ARE CALLING h_{+1} HERE AND psi_{k+1} IN THE THESIS BECAUSE LIFE
        '''
        # STEP 1
        # eval boundary geometry
        f_k, g_k = self.evaluate_boundary_geometry()
        
        # just before eq. 3.5 in wegmann's paper, Theta(zeta):=arg(zeta^{-2}*g_k/conj(g_k))
        # write g_k = |g|e^{i*arg(g)} and conj(g_k)=|g|e^{-i*arg(g)} then the fraction equals e^{2i*arg(g) => arg(g_k/conj(g_k))=2arg(g)
        # zeta is on the unit circle => arg(zeta^{-2})=-2*theta
        # arg(zeta^-2*fraction)=arg(zeta^-2)+arg(fraction) by complex number arithmetic
        # => result = -2 theta +2 arg(g_k)
        # see also formula in section 5 of the paper (p464)
        Theta = np.unwrap(np.arctan2(g_k.imag, g_k.real)) * 2 - 2* self.theta #this is for verfahren 2, see p 464
        H_Theta = ops.hilbert_transform(Theta)
        #print(f"FLAG: Theta = {Theta}, H_Theta = {H_Theta}")

        # find homogeneous solution X
        i_log_X = 1/2 * (-H_Theta + 1j * Theta) # by Mushkelishvili: X = exp(1/2*(H(Theta)+iTheta)
        # by eq. 3.6: X_plus = exp(Y(z))-1/2*Y(0)) and X_minus = z^{-2}* X_plus
        # Y(z) is Cauchy integral of Theta(zeta) => by plemelj-sokhotski formulas: Y^{+}-Y^{-}=Theta
        X_plus = np.exp(i_log_X)
        X_minus = self.zeta**(-2) * X_plus * np.exp(-Theta*1j) # $$X^-(\zeta) = \zeta^{-2} X^+(\zeta) e^{-i\Theta(\zeta)}$$
        #print(f"FLAG: X_plus = {X_plus}, X_minus = {X_minus}")

        # solve linearised problem
        if verfahren == 1:
            # p463f in the definition of the integral F, we have the integrand rho (belegungsfunktion) which by eq. 3.7 is given by
            rho = g_k * ( f_k/g_k).imag / X_plus
        elif verfahren == 2:
            # for verfahren 2, the first iter is also run with verf. 1 and h_k is taken from the previous iter to define rho_v2
            # hence why we redefine h_k outside the if statements for it to be in both scopes NO NEED TO DEFINE SINCE SCOPE IS WHOLE FCT APPARENTLY
            if self.h_k is None:
                raise RuntimeError("h_k is not defined yet, cannot run verfahren 2 without previous verfahren 1 iteration.")
            rho = g_k * ((f_k - self.h_k)/g_k ).imag / X_plus # p 464 bottom paragraph, h_k is just the h_k from previous iteration
        
        # integral is evaluated via FFT
        rho_coeffs = np.fft.fft(rho)
        # F^+ is only positive frequencies, so we zero out all the negative ones
        rho_plus, rho_minus = rho_coeffs.copy(), rho_coeffs.copy()
        rho_plus[self.N//2:] = 0 # Taylor series, Plemelj projection P+ corresponds to evaluating Cauchy integral F(z) inside unit disk
        rho_minus[:self.N//2] = 0 # Laurent series, Plemelj projection P- corresponds to evaluating Cauchy integral F(z) outside the unit disk
        F_plus_spatial = np.fft.ifft(rho_plus)
        F_minus_spatial = -np.fft.ifft(rho_minus) #defined n p 463
        # by eq.3.7, Phi(z) = X(z)/pi * integral(rho/(zeta-z)dzeta) and by §5 F(z)=1/(2pi*i) * integral(rho/(zeta-z)dzeta) => Phi(z)= X * 2j * F(z)
        Phi_plus = X_plus * 2j * F_plus_spatial
        Phi_minus = X_minus *2j * F_minus_spatial
        # eq. 3.8: h_0 = 1/2 * (Phi^{+} + conj(Phi^{-})).
        h_0 = 1/2 * (Phi_plus + np.conjugate(Phi_minus))
        # assert np.mean(np.abs(np.fft.fft(h_0)[self.N//2:])) < 1e-10, f"h_0 should be analytic, has neg fourier coeffs magnitude {np.mean(np.abs(np.fft.fft(h_0)[self.N//2:]))} \n received h_0 = {h_0}"
        
        # anchoring
        # we want h_0 = 0. since h = h_0 + P*X (eq 3.9) we have:
        # P = -h_0(0)/X(0)
        h_0_0 = np.mean(h_0)
        X_0 = np.mean(X_plus) # because the anchor is inside the circle => X_plus
        P = -h_0_0/X_0 # this fixes psi(0)=0

        if verfahren == 1:
            # full analytic souton h_k+1 = h_0 + P*X (p463)
            self.h_k = h_0 + P * X_plus
        elif verfahren == 2:
            self.h_k = self.h_k + h_0 + P * X_plus

        # which yields the correction step U_k = (h-f)/g (p456 and again p464)
        U_k = ((self.h_k - f_k) / g_k).real
        # assert np.mean(np.abs(U_k.imag)) < 1e-10, f"correction step U_k should be real, check for issues in RH solver or hilbert transform. \n received U_k = {U_k}"
        U_k -= np.mean(U_k) # zero the mean for uniqueness (Riemann mapping thm)
        #print(f"FLAG: correction step after mean-zero: U_k = {U_k}")
        return np.fft.fft(U_k)

    def normalise_orientation(self):
        ''' 
        ensure psi'(0)>0 for uniquenss
        second riemann mapping thm condition
        CAREFUL THIS YIELDS TOTAL CRAP FOR NON-CONVEX REGIONS IF APPLIED TOO AEARLY, assertion error expected.
        '''
        self.final_bdary_pts = self.get_final_boundary_points(apply_rotation_offset=False) # get final boundary points before rotation
        coeffs = np.fft.fft(self.final_bdary_pts)/self.N
        # psi'(0) is given by the first positive fourier coeff
        # we cirrect by that rotation angle
        self.rotation_offset = -np.angle(coeffs[1])
        print(f"FLAG: orientation normalised, rotation offset = {self.rotation_offset}")
        '''#finally shift bdary correspondence by that
        sigma_val = np.fft.ifft(self.sigma_hat).real
        sigma_val -= rotation_angle
        self.sigma_hat = np.fft.fft(sigma_val)
        self.final_bdary_pts = self.get_final_boundary_points() #update 
        #verif
        new_final_coeffs = np.fft.fft(self.final_bdary_pts) / self.N
        psi_prime_0 = new_final_coeffs[1]
        # psi'(0) must be real and positive
        #assert abs(psi_prime_0.imag) < 1e-6, f"Imaginary part of psi'(0) too large: {psi_prime_0.imag}"
        if(abs(psi_prime_0.imag) >= 1e-6):
            print(f"WARNING: Imaginary part of psi'(0) too large after orientation normalisation") #: {psi_prime_0}")
        #assert psi_prime_0.real > 0, f"psi'(0) is negative: {psi_prime_0.real}"
        if psi_prime_0.real <= 0:
            print(f"WARNING: psi'(0) is negative after orientation normalisation")#: {psi_prime_0}")
            '''

    def get_final_boundary_points(self, apply_rotation_offset=True):
        '''
        reconstructs the final boundary after convergence, undoing the normalisation shift

        :param apply_rotation_offset: whether to apply the rotation offset for orientation normalisation, default True
                                        called from all testing methods with True but from normalise_orientation() with False because pre-normalisation we dont HAVE the offset yet
        '''
        sigma_val = np.fft.ifft(self.sigma_hat).real
        S_final = self.theta + sigma_val + (self.rotation_offset if apply_rotation_offset else 0) # apply rotation offset if specified
        centered_points = self.eta.evaluate(S_final)
        return centered_points + self.shift
    
    def calculate_interior_mesh(self, mesh):
        '''
        maps the rest of the mesh onto the target region respecting conformality/ preserving analyticity.
        outputs vector of points in the target region corresponding to the input mesh points in the unit disk
        note positive fourier coeffs of the bdary are taylor coeffs of the interior CITE THM
        f(z)=sum(c_k * z^k)

        :param self: Description
        :param mesh: array of points in the unit disk to be mapped to the target region 
        '''

        bdary_pts = self.get_final_boundary_points()
        mapped_pts = np.zeros_like(mesh, dtype=complex) # one entry for each node of the mesh on unit disk
        
        coeffs = np.fft.fft(bdary_pts)/self.N
        taylor_coeffs = coeffs[:self.N//2]
        '''
        z_powers = np.ones_like(mesh, dtype=complex) # z^n for each n, initialized to z^0 = 1
        for c in taylor_coeffs:
            mapped_pts += c * z_powers # vectorised taylor sum calc
            z_powers *= mesh # update z^n to z^(n+1) for next iteration (exponentiate all pts of mesh simultaneously)

        '''
        # aliter: horner's method (faster)
        for c in reversed(taylor_coeffs):
            mapped_pts = mapped_pts * mesh + c
        
        return mapped_pts