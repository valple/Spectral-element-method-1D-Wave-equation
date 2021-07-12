import numpy as np
from numpy.polynomial import Legendre
from ricker import ricker

# Gauss-Legendre-Lobatto
# gll(n) gives n+1 points & weights
def gll(n):
    """Calculate Gauss-Legendre-Lobatto points and weights"""
    if n <2:
        raise ValueError("N must be at least 2") 
    a = np.zeros(n+1) # not very elegant
    a[n] = 1
    legpol = Legendre(a) # Legendre polynomial of order n-1
    x = np.zeros(n+1)
    x[0] = -1.
    x[n] = 1.
    x[1:n] = legpol.deriv().roots()
    w = np.zeros(n+1)
    w[0] = w[n] = 2/((n+1)*(n)) # boundary values
    w[1:n] = np.vectorize(lambda y: 2/((n+1)*(n)*((legpol(y))**2)))(x[1:n])
    return x, w

# First derivative of lagrange polynomials
# 
def lag_deriv(x, j, xi):
    """First derivative of j-th lagrange basis polynomial for points xi evaluated at x"""
    sum = 0
    n = len(xi)
    for i in range(0,n):
        fac = 1
        if i != j:
            for k in range(0,n):
                if k !=i and k !=j:
                    fac = fac * (x-xi[k])/(xi[j]-xi[k])
            fac = fac * 1/(xi[j]- xi[i])
            sum = sum + fac
    return sum
