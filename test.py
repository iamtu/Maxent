"""Trivial example: minimize x**2 from any start value"""

import lbfgs
import sys
from scipy.optimize import fmin_l_bfgs_b 
import numpy as np
class Test:
    def __init__(self):
        self.constant = 10
        self.X = np.zeros(2)
    def f(self, X):
        f_value = (X[0]-2) ** 2 + (X[1]-1) ** 2 + self.constant
        grad = np.zeros(2)
        grad[0] = 2 * (X[0]-2)
        grad[1] = 2 * (X[1]-1)
        return f_value, grad
    def grad(self, X):
        a = np.zeros(2)
        a[0] = 2 * (X[0]-2)
        a[1] = 2 * (X[1]-1)
        return a
    def train(self):
        self.X, f , d=  fmin_l_bfgs_b(self.f, self.X, fprime=None , iprint = 99)
        print 'X_value', self.X
        print 'final value', f
        
if __name__ == '__main__':
    test = Test()
    test.train()