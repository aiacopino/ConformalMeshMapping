'''
from targetregion import BoundaryCurve
## wegmann's method algorithm for conformal mapping
input: boundary parametrisation eta of target region, anchoring points z_0 on unit disk, s_0 in [0,2pi]
output: conformal map psi from unit disk to target region


class WegmannSolver:
        #perform one newton iter
    def step():
        # eval boundary geometry
        f = eta.evaluate(s)
        Df = eta.derivative(s)
        # RH problem
        RHS = (f/Df).imag
        LHS = None
        pass
    def solve():
        #iterate until convergenceorget about energy management and only focus on solving 
        pass
    def count_iterations():
        pass

## initial guess for boundary correspondence function S

## iterated correction

## calculate fourier coefficients

## new iterate

## make entire mesh from boundary

## count iterations/ convergence rate
'''