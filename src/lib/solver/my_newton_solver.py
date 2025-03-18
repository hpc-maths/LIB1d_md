# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:21:35 2022

@author: J1043337
"""
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy import linalg
import scipy.sparse as sparse

# Custom newton solver to solve the non-linear system formed buy the integral
def root_solve(func,
               x0,
               verbose=None,
               sparsity=None,
               atol=None,
               rtol=None,
               freq=None,
               deBug=False,
               args=()):
    if not isinstance(args, tuple):
        args = (args,)
    ftol = 1e-15 
    itr = 0
    xk = x0
    xkk = xk
    
    if verbose is None:
        verbose=False
        
    if atol is None:
        atol= 1e-10
    
    if rtol is None:
        rtol = 1e-9
    
    if freq is None:
        freq = 1
    
    if sparsity is None:
        sparse_solve=False
    else:
        sparse_solve=True
        
    itr_max = 15
    
    while (1): 
        if(itr%freq==0):
            if(sparse_solve):
                Jk = approx_derivative(func, xk, method='2-point',
                                       sparsity=sparsity,
                                       rel_step=1e-8, abs_step=1e-8,
                                       args=args)
            else:
                Jk = approx_derivative(func, xk, args=args)
                
        fk = func(xk)
        
        if(sparse_solve):
            delk = sparse.linalg.spsolve(Jk, -fk)
        else:
            delk = linalg.solve(Jk, -fk)
            
                
        xkk = xk + delk
        itr+=1
        if( itr >= itr_max):
            xk = xkk
            raise Exception(f"Newton solver reached maximum iterations of {itr_max}")
            break
        else:
            fkk = func(xkk)
            if deBug:
                print(f"debug: residual = {linalg.norm(fkk):.2e}")
                print(f"debug: error = {linalg.norm(np.abs(xkk-xk)/(rtol*np.abs(xk) + atol)):.2e}")
            if(np.isnan(xkk).any() or np.isinf(xkk).any() or np.isnan(fkk).any() or np.isinf(fkk).any()):
                raise Exception("Nans appeared in solution")
                break
            if(linalg.norm(np.abs(xkk-xk)/(rtol*np.abs(xk) + atol)) <= 1.0 or linalg.norm(fkk) <= ftol):
                xk = xkk
                break
        xk = xkk
    
    if(verbose):
        print("No. of Newton iterations = ", itr)
    return xk
