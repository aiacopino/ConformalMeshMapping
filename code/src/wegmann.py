import numpy as np
from scipy.fft import fft, ifft
import src.operators as ops
from src.targetregion import BoundaryCurve
import functools #maybe delete later

class WegmannSolver:
    '''
    wegmann's method algorithm for conformal mapping
    input: boundary parametrisation eta of target region, anchoring points z_0 on unit disk, s_0 in [0,2pi]
    output: conformal map psi from unit disk to target region
    all fourier coeffs are in 2xN matrices with row 0 = real part, row 1 = imaginary part
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
        centered_coeffs = eta.coeffs.copy()
        centered_coeffs[0] = 0
        self.eta = BoundaryCurve(centered_coeffs)

        if len(self.eta.coeffs) % 2 != 0: #check even number of fourier coeffs
            raise ValueError("WegmannSolver __init__: Number of Fourier coefficients for boundary parametrisation eta must be even.")

        self.theta = np.linspace(0, 2*np.pi, num=N, endpoint=False) # array of equally spaced args (angles)
        self.zeta = np.exp(1j*self.theta) # array of equally spaced grid pts on unit circle [0, 2pi)

        # fix normalization psi(0) = 0, psi'(0) > 0 for uniqueness (anchoring)
        # what if normalization point is somewhere else: make adjustable to user input
        # self.z_0 = 0 # eta.evaluate(0) ################### check this ######################
        # self.s_0 = 0
        
        #state variables
        self.sigma_hat = None # fourier coeffs of boundary correspondenec fct, equation 33 in thesis (fourier coeffs of sigma)
        self.error_history = [] # convergence tracking

    
    def init_initial_guess(self):
        '''
        initial guess for boundary correspondence function S mapping angle theta (in [0, 2pi]) of unit disk to boundary parameter s of target region (eta(S))        
        '''
        # initial guess is identity map S(theta) = theta
        # S(theta) = theta + sigma(theta) where sigma is 2pi-periodic smooth (equation 33 in thesis)
        # hence sigma = 0 is initial guess
        self.sigma_hat = np.zeros(self.N, dtype=complex) 

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
        sigma_dot_hat = ops.fourier_derivative(self.sigma_hat)
        sigma_dot_val = np.fft.ifft(sigma_dot_hat).real
        S_dot_k = 1 + sigma_dot_val
        
        f_k = self.eta.evaluate(S_k) #position on boundary f_k = η(S_k(θ)), geometric candidate but not analytic -> not extendable into interior of disk (analytic candidate is psi_k+1 on the tangent)
        #g_hat = ops.fourier_derivative(f_k) * S_dot_k #tangent on buondary g_k = η'(S_k(θ)) * S_k'(θ)
        g_k = self.eta.evaluate_derivative(S_k) * S_dot_k
        return f_k, g_k

    def compute_U_k_correction_hat(self, psi, f_k, g_k):
        '''
        compute fourier coeffs of correction U_k where S_k+1 = S_k + U_k        
        :param psi: approximation of conformal map on tangent
        :param f_k: approximation of conformal map on boundary
        :param g_k: tangent vector at boundary
        '''
        U_k_correction = ((psi - f_k) / g_k).real
        return np.fft.fft(U_k_correction)
    
    def compute_new_sigma_hat(self, U_K_correction_hat):
        '''
        update parametrisation parameter sigma using the correction U_k computed from the RH problem solution
        $S_{k+1}(theta) = S_k(theta) + U_k(theta)$
        where $S(theta) = theta + sigma(theta)$
        thus, theta + sigma_{k+1} = theta + sigma_k + U_k

        '''
        return self.sigma_hat + U_K_correction_hat
    
    def detrended_periodic_A(self, angle_A):
        '''
        helper function to mitigate gibbs phenomenon in hilbert transform of angle_A when computing M for RH problem solution
        alternative to just hilbert transforming the nagle directly

        decomposes signal u(t) = periodic_u(t) + m*t
        H[u] = H[periodic_u] + H[m*t]
        '''
        N = len(angle_A)
        #calc linear drift; is zero if the array is already periodic 
        # -> no change if the input is already non-pathological
        drift = angle_A[-1] - angle_A[0] # numerical noise
        #subtract drift to make it periodic
        slope_vec = np.linspace(0, drift, N)
        print(f"FLAG: detrending hilbert input, drift = {drift}")
        periodic_part = angle_A - slope_vec
        return periodic_part, slope_vec

    def solve_RH_problem(self, A, RHS):
        # step 2 of algorithm 3 in thesis
        # https://arxiv.org/abs/1210.2199
        '''
        Solves Re( psi / A ) = RHS where Index(A) = 0: psi = M * (Re(W) + i*H[Re(W)])
        homogeneous solution Ansatz: psi = M * W where M solves the homogeneous problem Re(M/A)=0, and W is the unknown function we solve for in the reduced Dirichlet problem
        M = multiplier (homogeneous solution) => M solves Re(psi/A)=0
        usually integrating factor of RH problem is multiplied to achieve a winding number of 0 (=> real)
        W := psi/M is defined in order to reduce to Dirichlet problem: since M/A real (because pahse(M)=phase(A) => imaginary part divides out),
        Re(M*W/A)=RHS <=> M/A* Re(W) = RHS <=> Re(W) = RHS * A/M
        then the imaginary part of W is recovered via hilbert transform
        finally psi = M*W
        -
        -
        :param A: coefficient function in RH problem, array of complex values on unit circle
        :param RHS: right hand side of RH problem, array of real values on unit circle
        '''
        # unwrap arg
        angle_A = np.angle(A)
        angle_A = np.unwrap(angle_A) # smoothing discontinuities coming from modulo 2pi
        print(f"FLAG: max jump in angle of A (should be small after unwrapping): {np.max(np.abs(np.diff(angle_A)))}")
        # M = exp( H[angle] + i*angle )
        #H_angle = ops.hilbert_transform(angle_A) # this creates gibbs phenomenon
        #M = np.exp(H_angle + 1j * angle_A) #this makes the error from gibbs phenomenon astronomical
        #maybe next time implementation of nasser's work using second order fredholm equations and generalized neumann kernel would be better
        # as it allows exploitation of the compact operator and is not affected by gibbs phenomenon
        # alternatively, we introduce a helper function to work around the gibbs phenomenon by detrending the angle signal before hilbert transforming it, then reintroducing the trend after the transform 
        # (since the trend is linear, its hilbert transform is known analytically and can be added back in easily)
        angle_A_detrended, slope = self.detrended_periodic_A(angle_A)
        #transform only periodic part
        H_angle = ops.hilbert_transform(angle_A_detrended)
        H_slope = ops.hilbert_transform(slope)
        H_full = H_angle + H_slope
        print(f"FLAG: max and min of H[angle_A] (after detrended hilbert transformation): {np.max(H_angle)}, {np.min(H_angle)}. H(slope)= {H_slope}")
        M = np.exp(H_full + 1j * angle_A)
        # Check ratio (M/A should be real)
        ratio = (M / A).real 
        print(f"FLAG: RH problem check, max imag part of M/A: {np.max(np.abs((M/A).imag))} (should be close to 0)")

        # Solve for W = phi/M
        Re_W = RHS / ratio
        Im_W = ops.hilbert_transform(Re_W) # normalisation with mean = 0 (riemann mapping thm)
        
        W = Re_W + 1j * Im_W
        
        return M * W

    def newton_step(self):
        '''
        perform one newton iteration
        stuff within the while loop (lines 4-11) of algorithm 3 in thesis
        '''
        # STEP 1
        # eval boundary geometry
        f_k, g_k = self.evaluate_boundary_geometry()

        # STEP 2
        # solve RH problem
        # RHS = Im (f(zeta)/tangent_vec(zeta))
        RHS = (f_k/g_k).imag # tangent vec has winding number 1, need winding number 0 for std RH solver
        # g_k is tangent vector -> 1j*g_k is normal vector
        A = 1j*g_k / self.zeta #new coefficient with winding nr 0
        # RH solves Re(psi_bar/A) = RHS
        psi_bar = self.solve_RH_problem(A, RHS)

        # recover psi from psi_bar undoing the division by zeta in A 
        # also this fixes degree of freedom f(0)=0 required for uniqueness by riemann mapping thm
        # this crashes if target is not centered at 0 -> make adjustable
        psi = psi_bar * self.zeta

        # STEP 3
        # compute correction 
        U_k_correction_hat = self.compute_U_k_correction_hat(psi, f_k, g_k)
        
        # STEP 4
        # update sigma_hat using correction U_k

        return U_k_correction_hat
    

    def find_conformal_map(self, max_iter = 100, epsilon = 1e-6):
        '''
        Wegmann's method for conformal mapping solver        
        :param self: Description
        :param max_iter: maximum iterations of Newton step before stopping
        :param epsilon: convergence tolerance for stopping criterion
        '''
        # damping factor to stabilize convergence if initial guess is way off
        # if convergence is too slow, this can be increased
        relaxation = 0.7

        self.init_initial_guess()
        for i in range(max_iter):
            old_sigma = self.sigma_hat.copy()
            U_k_hat = self.newton_step()

            # -------- anti divergence: low-pass filter (orszag's 2/3 rule)
            # zero out the top 1/3 of freqencies to avoid accumulation of high-freq noise
            freqs = np.fft.fftfreq(self.N) #ordered: [0,1,2,...,N/2, -N/2, ..., -2, -1] 
            cutoff_fraction = 1/3 # 0.25 MORE STABLE 2/3 * 0.5 (Nyquist frequency) "cpt support" like enforcing Schwarz decay"
            high_freq_mask = np.abs(freqs) > cutoff_fraction
            print(f"FAG: applying low-pass filter to correction, zeroing out {np.sum(high_freq_mask)} out of {self.N} frequencies :\n {freqs[high_freq_mask]}")
            #apply filter: zero out entries where the mask applies
            U_k_hat [high_freq_mask] = 0.0 + 0.0j #can i also just write 0 here?

            #self.sigma_hat = old_sigma + relaxation * U_k_hat # sigma_new = sigma_old + U_k (by equation 33) this diverges wildly
            self.sigma_hat = old_sigma - relaxation * U_k_hat # now at least the error steadily decreases, provbably an issue of boundary orientation
            
            #enforce mean for uniqueness (riemann mapping thm); more numerically stable option than pinning a given point which could be badly chosen
            # change this to anchoring a specific point for non-symmetric target shapes
            self.sigma_hat[0] = 0 

            # if U_k is very small, we are done (remaining correction is very small, algorithm "settled" -> converged)
            # coul we also do this measuring the output of compute_U_k_correction_hat() fct?
            err = np.linalg.norm(self.sigma_hat - old_sigma) # norm of the correction vector
            if err > 1e3:
                print(f"ERR: solver diverging, stopping. Last error: {err} iteration: {i}")
                break
            self.error_history.append(err)
            print(f"FLAG: iter {i}, error: {err}")

            #check if converged yet
            if err < epsilon:
                print(f"Converged in {i} iterations.")# \n Error history: {self.error_history}")
                return self.sigma_hat
        #if not converged after max_iter
        print(f"Max iterations reached. \n Error history: {self.error_history}")

        return self.sigma_hat
    
    def get_final_boundary_points(self):
        '''
        reconstructs the final boundary after convergence, undoing the normalisation shift
        '''
        sigma_val = np.fft.ifft(self.sigma_hat).real
        S_final = self.theta + sigma_val
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
        CARTESIAN OR RADIAL? tbd
        '''

        bdary_pts = self.get_final_boundary_points()
        mapped_pts = np.zeros_like(mesh, dtype=complex) # one entry for each node of the mesh on unit disk
        
        coeffs = np.fft.fft(bdary_pts)/self.N
        taylor_coeffs = coeffs[:self.N//2]
        z_powers = np.ones_like(mesh, dtype=complex) # z^n for each n, initialized to z^0 = 1
        for c in taylor_coeffs:
            mapped_pts += c * z_powers # vectorised taylor sum calc
            z_powers *= mesh # update z^n to z^(n+1) for next iteration (exponentiate all pts of mesh simultaneously)

        '''
        # aliter: horner's method (faster)
        for c in reversed(taylor_coeffs):
            mapped_pts = mapped_pts * mesh + c
        '''
        '''
        for p in mesh:
            # compute cauchy integral formula for each interior point
            # psi(p) = 1/(2pi i) * int_{boundary} (psi(t) / (t-p)) dt
            # where psi(t) is the boundary value at t, and the integral is taken over the boundary curve
            # this should give us the value of the conformal map at the interior point pt
            # vectorize this
            integrand = bdary_pts / (self.zeta - p)
            mapped_pts += np.sum(integrand) / (2 * np.pi * 1j) # approximate integral by sum over discretized bdary pts
            # does this make sense tho? shold this be computed for each interior mesh point or can we do all at once somehow?
            # also is this sum correct or should it be append or like mapped_pts[i] = sum(integrand) / (2 * np.pi * 1j) for each i? since we already allocated the storage for mapped_pts
        '''
        return mapped_pts