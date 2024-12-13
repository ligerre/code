import numpy as np
import math
np.seterr(divide='ignore', invalid='ignore')
def proj_GD_Polyak(oracle,proj,projN,x0,niters,op_val):
    """projection GD with Polyak stepsize"""
    tol = 10**(-9)
    x_k = np.copy(x0)
    res = []
    grad = []
    step_size = []
  
    for i in range(niters):
        value, grad_x = oracle(x_k)
        res.append(value)
        grad_norm = np.linalg.norm(grad_x- projN(x_k,grad_x))
        grad.append(grad_norm)
        if grad_norm < tol:
            break

        eta = (value - op_val)/((np.linalg.norm(grad_x)**2))
        step_size.append(eta)
      
        x_k = proj(x_k-eta*grad_x)
    return res,grad,step_size,x_k
def NCGD_Polyak(oracle,proj,projN,x0,niters,op_val):
    """projection GD with Polyak stepsize and taking tangential component"""
    tol = 10**(-9)
    x_k = np.copy(x0)
    res = []
    grad = []
    step_size = []
    points = []
    for i in range(niters):
        value, grad_x = oracle(x_k)
        
        grad_x = grad_x - projN(x_k,grad_x)
        grad_norm = np.linalg.norm(grad_x)
        res.append(value)
        points.append(x_k)
        grad.append(grad_norm)
        if grad_norm < tol:
            break
       # if np.linalg.norm(grad_x-prev_grad)<=tol:
        #    break
        eta = (value-op_val)/((grad_norm **2))         
        step_size.append(eta)
        x_k = proj(x_k-eta*grad_x)

    
    return res,grad,step_size,points
def proj_GD(oracle,proj,projN,x0,niters,lr=0.001):
    """projection GD with Professor Malitsky Adaptive stepsize"""
    tol = 10**(-9)
    x_k = np.copy(x0)
    res = []
    grad = []
    step_size = []
    eta = lr
    theta = 100000
    prev_x = np.copy(x_k)
    value, grad_x = oracle(x_k)
    res.append(value)
    grad.append(np.linalg.norm(grad_x- projN(x_k,grad_x)))
    step_size.append(eta)
    x_k = proj(x_k-eta*grad_x)
    
    for i in range(niters):
        prev_grad = np.copy(grad_x)
        

        value, grad_x = oracle(x_k)
        res.append(value)
        grad_norm = np.linalg.norm(grad_x- projN(x_k,grad_x))
        grad.append(grad_norm)
        if grad_norm < tol:
            break
       # if np.linalg.norm(grad_x-prev_grad)<=tol:
        #    break
        prev_eta = eta
        try:
            eta = min(np.sqrt(1+theta)*prev_eta,np.linalg.norm(x_k-prev_x)/(2*np.linalg.norm(grad_x-prev_grad)))
        except ZeroDivisionError:
            eta = np.sqrt(1+theta)*prev_eta
        step_size.append(eta)
        
        prev_x = np.copy(x_k)
        x_k = proj(x_k-eta*grad_x)

        theta = eta/prev_eta
    
    return res,grad,step_size,x_k

def NCGD(oracle,proj,projN,x0,niters,lr=0.001):
    """projection GD with Professor Malitsky Adaptive stepsize and taking tangential component"""
    tol = 10**(-9)
    x_k = np.copy(x0)
    res = []
    grad = []
    step_size = []
    points = []
    eta = lr
    theta = 100000
    prev_x = np.copy(x_k)

    value, grad_x = oracle(x_k)
    points.append(x_k)
    res.append(value)
    grad_x = grad_x - projN(x_k,grad_x)
    grad.append(np.linalg.norm(grad_x))
    x_k = proj(x_k-eta*grad_x)
    step_size.append(eta)
    for i in range(niters):
        prev_grad = np.copy(grad_x)
        

        value, grad_x = oracle(x_k)
        
        grad_x = grad_x - projN(x_k,grad_x)
        grad_norm = np.linalg.norm(grad_x)
        res.append(value)
        points.append(x_k)
        grad.append(grad_norm)
        if grad_norm < tol:
            break
       # if np.linalg.norm(grad_x-prev_grad)<=tol:
        #    break
        prev_eta = eta
        try:
            eta = min(np.sqrt(1+theta)*prev_eta,np.linalg.norm(x_k-prev_x)/(2*np.linalg.norm(grad_x-prev_grad)))
        except ZeroDivisionError:
            eta = np.sqrt(1+theta)*prev_eta
        step_size.append(eta)
        
        
        prev_x = np.copy(x_k)
        x_k = proj(x_k-eta*grad_x)

        theta = eta/prev_eta
    
    return res,grad,step_size,points