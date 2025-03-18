#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:08:08 2022

@author: ali.asad
"""

import numpy as np
import scipy.sparse as sparse

# from src.lib.reader import  reader as params
from src.lib.solver import my_newton_solver as newton

####################################################################################################################################################

# Backward-Euler solver for time integration of a Index 1 DAE-system
class dae_result:
    def __init__(self, y, t):
        self.y = y
        self.t = t

def integrate_dae(mass,
                  tini,
                  tend,
                  nt,
                  yini,
                  fcn,
                  options_electrolyte,
                  options_cathode,
                  rtol=None,
                  atol=None,
                  verbose=None,
                  verbose_freq=None):
        
    # Checking parameters and default setting
    if(verbose is None):
        verbose = False
        
    if(verbose_freq is None):
        verbose_freq = 1
    
    if atol is None:
        atol = 1e-8
    
    if rtol is None:
        if atol is not None:
            rtol = atol*10.
        else:
            rtol = 1e-7
        
    # Initiating parameters for time integration
    dt = (tend - tini) / (nt-1)
    t = np.linspace(tini, tend, nt)
    # params.sim_time = t[0]*params.t_c

    yini = np.atleast_1d(yini) # atleast_1d --> converts the value to at least a 1D array eg: 1.0 becomes np.array(1.0)
    neq = yini.size

    y = np.zeros((neq, nt), order='F') # store in fortran-style column major
    y[:,0] = yini

    # sparsity pattern
    uband=6; lband=-uband
    offsets = [i for i in range(lband,uband)]
    sparsity_pattern = sparse.diags(diagonals=[np.ones((neq - abs(i))) for i in offsets], offsets=offsets)
    

    def residual(uip1, uip, tip1):
        return np.dot(mass, (uip1 - uip)) - dt*np.array(fcn (uip1, tip1))

    print("---------------------------------- \nTime integration begins")
    for it, tn  in enumerate(t[:-1]):
        yn = y[:,it]
        
        if(verbose):
            if(it%verbose_freq==0):
                print(f"t = {t[it]*options_electrolyte['parameters']['t_c']:.2f} s \t \t V = {y[-1, it]*options_electrolyte['parameters']['phi_c']:.8f} volts")
        
        solx = newton.root_solve(lambda x: residual(uip1=x,
                                             uip=yn,
                                             tip1=tn+dt),
                                             yn, verbose, sparsity_pattern,
                                             atol=atol, rtol=rtol)
                                           
        y[:,it+1] = solx
#        params.time_sim = t[it+1]*params.t_c
        
        # if (params.time_sim >= 0.4*params.tend_dim and params.time_sim < 0.5*params.tend_dim):
        #     params.xi = 0.0
        # elif (params.time_sim >= 0.5*params.tend_dim): 
        #     params.xi = -0.2
                
    if(verbose):
        print(f"t = {t[-1]*options_electrolyte['parameters']['t_c']:.2f} s \t \t V = {y[-1, -1]*options_electrolyte['parameters']['phi_c']:.8f} volts")
        print("Simulation completed")
        print("--------------------------------------------------------------------------------")
    else:
        print(".\n.\nSimulation completed")
        print(f"time = {t[-1]*options_electrolyte['parameters']['t_c']:.2f} s \t voltage = {y[-1, -1]*options_electrolyte['parameters']['phi_c']:.8f} volts")
        print("--------------------------------------------------------------------------------")
        
    return dae_result(y, t)