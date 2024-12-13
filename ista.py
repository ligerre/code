from math import sqrt
import numpy as np
from scipy import linalg

def ista(x0,A,b,l,niters=5000):
    res = []
    x_k = np.copy(x0)
    L = linalg.norm(A) ** 2  # Lipschitz constant
    for _ in range(niters):
        val,grad = oracle(A,x_k,b,l)
        res.append(val)
        x_k = soft_thresh(x_k-grad/L,l/L)

    return res

def ista2(x0,A,b,l,niters=5000):
    res = []
    x_k = np.copy(x0)
    L = linalg.norm(A) ** 2  # Lipschitz constant
    for _ in range(niters):
        val,grad = oracle(A,x_k,b,l)
        res.append(val)
        grad_n = grad - subgradient_proj(x_k,grad,l)
        x_k = soft_thresh(x_k-grad_n/L,l/L)

    return res