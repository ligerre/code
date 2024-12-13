import numpy as np

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)
def subgradient_proj(x,y,l):
    """Project y into subgradient of l|x|"""
    out = np.zeros_like(y)

    idx0 = x != 0
    out[idx0] = -l*np.sign(x[idx0])

    idx1 = x==0
    """
    idy1 = np.abs(y)<l
    idy2 = np.abs(y)>=l

    id1 = np.logical_and(idx1,idy1)
    out[id1] = y[id1]
    
    id2 = np.logical_and(idx1,idy2)
    out[id2] = l*np.sign(y[id2])
    """
    out[idx1] = np.sign(y[idx1])*np.minimum(np.abs(y[idx1]),l)
    return out