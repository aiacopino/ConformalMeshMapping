import numpy as np
import src.operators as ops
from targetregion import BoundaryCurve
import functools


class WegmannSolver:
    '''
    wegmann's method algorithm for conformal mapping
    input: boundary parametrisation eta of target region, anchoring points z_0 on unit disk, s_0 in [0,2pi]
    output: conformal map psi from unit disk to target region
    '''
    def __init__(self, eta, N):
        self.eta = eta
        self.N = N

        # fix normalization psi(0) = eta(s_0) for uniqueness
        ################### check this ######################
        self.z_0 = eta.evaluate(0)
        self.s_0 = 0

        #state variables
        self.current_approximation_coeffs = None
        self.current_derivative_coeffs = None
        self.error_history = [] # convergence tracking

    def init_initial_guess(self):
        pass


    def step(self):
        #perform one newton iteration
        # eval boundary geometry
        f = self.eta
        g = ops.fourier_derivative(f)
        # RH problem
        RHS = (f/g).imag
        LHS = None
        pass

    def solve(self, max_iter = 100, epsilon = 1e-6):
        #iterate until convergence
        self.init_initial_guess()
        for i in range(max_iter):
            err = self.step()
            self.error_history.append(err)
            if err < epsilon:
                print(f"Converged in {i} iterations. \n Error history: {self.error_history}")
                return self.current_approximation_coeffs
        print(f"Max iterations reached. \n Error history: {self.error_history}")
        return self.current_approximation_coeffs

    def count_iterations():
        pass

    @functools.cached_property
    def current_tangent_angle(self):
        # calc current tangent angle from current derivative coeffs
        # cached property so only calculated once per iteration

        pass

## initial guess for boundary correspondence function S

## iterated correction

## calculate fourier coefficients

## new iterate

## make entire mesh from boundary

## count iterations/ convergence rate