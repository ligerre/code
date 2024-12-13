import cvxpy as cvx
import numpy as np
"""For probability simplex"""
def proj_simplex(y):
    """project down to the probability simplex"""
    m=len(y)
    bget = False
    
    s = sorted(np.array(y), reverse=True)
    tmpsum = 0.0

    for ii in range(m-1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1.0) / (ii+1.0)
        if tmax >= s[ii+1]:
            bget = True
            break

    if bget == False:
        tmax = (tmpsum + s[m-1] - 1.0) / m

    x = np.maximum(np.array(y) - tmax, np.zeros(m))

    return x
def active_set(x):
    tol = 10**(-12)
    d = len(x)
    activ = np.zeros((d,d))
    val = -1
    for i in range(d):
        if x[i]>tol:
            val = i
            activ[i][i]=1
    return activ,val
def proj_simplex_normal(z,y):
    '''
    return projection onto negative normal cone of simplex
    '''
    activ,val = active_set(z)
    d = len(y)
    x = cvx.Variable(d)
    obj = cvx.Minimize(cvx.sum_squares(x - y))
    if val == -1:
        constr = [x<=0]
    else:
        slack = x[val]*np.ones(d)
        constr = [x>=slack, activ@x == activ@slack]
    prob = cvx.Problem(obj, constr)
    prob.solve()
 
    return np.array(x.value).squeeze()

"""For unit ball"""

def proj_unit(x):
    """project down to unit ball"""
    if np.linalg.norm(x)>1:
        return x/np.linalg.norm(x)
    else:
        return x

def proj_unit_sphere(x):
    """project down to unit sphrere"""
    if np.linalg.norm(x)==0:
        x[0]=1
        return x
    return x/np.linalg.norm(x)
 
def proj_unit_normal(x,y):
    """return projection of y to negative normal cone of x in unit sphere"""
    tol = 10**(-9)
    if (1-np.linalg.norm(x))>=tol:
        return np.zeros(len(y))
    else:
        return min(0,np.dot(x,y))*x
    

"""For positive quadrant"""

def proj_positive_quad(x):
    d = len(x)
    res = np.copy(x)
    for i in range(d):
        res[i]=max(0,x[i])
    return res

def proj_positive_quad_normal(x,y):
    z = np.copy(y)
    for i in range(len(x)):
        if x[i]>0:
            z[i]=0
        else:
            z[i]=max(0,y[i])
    return z

"""no projection"""
def proj_none(x):
    return x
def proj_normal_none(x,y):
    return np.zeros_like(y)