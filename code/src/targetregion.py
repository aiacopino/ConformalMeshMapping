import numpy as np
class BoundaryCurve:
    def __init__(self, matrix):
        rows, cols = matrix.shape
        self.coefficients = matrix
        self.N = cols

        if rows != 2:
            raise ValueError(f"Expected 2xN array for boundary coefficients, got {rows}x{cols}.")

    def evaluate(self, t):
        pass
    def derivative(self, t):
        pass