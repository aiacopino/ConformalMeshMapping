import numpy as np
from scipy.fft import fft, ifft
import src.operators as ops
from targetregion import BoundaryCurve
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
        :param self: Description
        :param eta: boundary parametrisation, format = array of fourier coefficients (complex)
        :param N: number of discretization points on the boundary
        '''
        self.eta = eta
        self.N = N

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

        #derivative of S_k : S_k = theta + sigma(theta) => S_k'(θ) = 1 + σ'(θ)
        sigma_dot_hat = ops.fourier_derivative(self.sigma_hat)
        sigma_dot_val = np.fft.ifft(sigma_dot_hat).real
        S_dot_k = 1 + sigma_dot_val

        f_k = self.eta.evaluate(S_k) #position on boundary f_k = η(S_k(θ)), geometric candidate but not analytic -> not extendable into interior of disk (analytic candidate is psi_k+1 on the tangent)
        g_k = ops.fourier_derivative(f_k) * S_dot_k #tangent on buondary g_k = η'(S_k(θ)) * S_k'(θ)

        return f_k, g_k

    def compute_U_k_correction_hat(psi, f_k, g_k):
        '''
        compute fourier coeffs of correction U_k where S_k+1 = S_k + U_k        
        :param psi: approximation of conformal map on tangent
        :param f_k: approximation of conformal map on boundary
        :param g_k: tangent vector at boundary
        '''
        U_k_correction = ((psi - f_k) / g_k).real
        return np.fft.fft(U_k_correction)
    
    def compute_new_sigma_hat():
        return None

    def solve_RH_problem(A, RHS):
        # step 2 of algorithm 3 in thesis
        # https://arxiv.org/abs/1210.2199
        '''
        Solves Re( psi / A ) = RHS where Index(A) = 0: psi = M * (Re(W) + i*H[Re(W)])
        M = multiplier (homogeneous solution) => M solves Re(psi/A)=0
        usually integrating factor of RH problem is multiplied to achieve a winding number of 0 (=> \R)
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
        
        # M = exp( H[angle] + i*angle )
        h_angle = ops.hilbert_transform(angle_A) # should i use standard (built-in) fct here instead of hand-written hilbert_transform?
        M = np.exp(h_angle + 1j * angle_A) # WHY IS IT PLUS HERE??
        
        # Check ratio (M/A should be real) WHAT IF IT ISNT? CHECK THIS DIFFERENTLY
        ratio = (M / A).real 
        print(f"FLAG: RH problem check, max imag part of M/A: {np.max(np.abs((M/A).imag))}") # should be close to 0

        # Solve for W = phi/M
        Re_W = RHS / ratio
        Im_W = ops.hilbert_transform(Re_W)
        
        W = Re_W + 1j * Im_W
        
        return M * W

    def step(self):
        '''
        perform one newton iteration
        stuff within the while loop (lines 4-12) of algorithm 3 in thesis
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
        psi_bar = self.RH_problem(A, RHS)
        psi = psi_bar * self.zeta # recover psi from psi_bar undoing the division by zeta in A

        # STEP 3
        #correction and update 
        U_k_correction_hat = self.compute_U_k_correction_hat(psi, f_k, g_k)

        # if U_k is very small, we are done (remaining correction is very small, algorithm "settled" -> converged)
        # should this be checked here or in solve() function??
        # does this belong here or in solve() fct? given it only needs to be done if we are not converged yet 
        # also, if the error really measures U_k this should be he output, not sigma, right?
        # update sigma
        new_sigma_hat = self.sigma_hat + U_k_correction_hat # sigma_new = sigma_old + U_k (by equation 33)
        new_sigma_hat[0] = 0 #enforce mean to keep S as a diffeo of the circle #MAYBE MAKE THIS ADAPTABLE/ USER INPUT DEPENDENT
        
        return new_sigma_hat
    

    def find_conformal_map(self, max_iter = 100, epsilon = 1e-6):
        '''
        Wegmann's method for conformal mapping solver        
        :param self: Description
        :param max_iter: maximum iterations of Newton step before stopping
        :param epsilon: convergence tolerance for stopping criterion
        '''
        self.init_initial_guess()
        for i in range(max_iter):
            old_sigma = self.sigma_hat.copy()
            self.sigma_hat = self.step()
            
            # if U_k is very small, we are done (remaining correction is very small, algorithm "settled" -> converged)
            # coul we also do this measuring the output of compute_U_k_correction_hat() fct?
            err = np.linalg.norm(self.sigma_hat - old_sigma) # norm of the correction vector
            self.error_history.append(err)
            print(f"iter {i}, error: {err}")

            #check if converged yet
            if err < epsilon:
                print(f"Converged in {i} iterations. \n Error history: {self.error_history}")
                return self.sigma_hat
        #if not converged after max_iter
        print(f"Max iterations reached. \n Error history: {self.error_history}")
        return self.sigma_hat #this is not the conformal map? output incorrect?
    

'''
    @functools.cached_property
    def current_tangent_angle(self):
        # calc current tangent angle from current derivative coeffs
        # cached property so only calculated once per iteration
        pass
'''