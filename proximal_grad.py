from math import sqrt
import numpy as np
from scipy import linalg

def ista(oracle,soft_thresh,x0,L,l,niters=5000):
    res = []
    gradient = []
    x_k = np.copy(x0)
    for _ in range(niters):
        val,grad = oracle(x_k,l)
        res.append(val)
        gradient.append(linalg.norm(grad))
        x_k = soft_thresh(x_k-grad/L,l/L)

    return res,gradient

def ista2(oracle,soft_thresh,subgradient_proj,x0,L,l,niters=5000):
    res = []
    gradient = []
    x_k = np.copy(x0)
    for _ in range(niters):
        val,grad = oracle(x_k,l)
        res.append(val)
        gradient.append(linalg.norm(grad))
        grad_n = grad - subgradient_proj(x_k,grad,l)
        x_k = soft_thresh(x_k-grad_n/L,l/L)

    return res, gradient