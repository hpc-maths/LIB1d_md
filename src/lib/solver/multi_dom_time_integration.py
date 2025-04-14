#!/usr/bin/env python3
# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:08:08 2022

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
                  fcn_e,
                  fcn_s,
                  options_electrolyte,
                  options_cathode,
                  c_tol=None,
                  max_c_itr=None,
                  sub_system_rtol=1e-6,
                  verbose_ss=None,
                  verbose_cter=None,
                  verbose_freq=None):
    
    # Checking parameters and default setting
    if(c_tol is None):
        c_tol = 5.0e-4
    
    if(max_c_itr is None):
        max_c_itr = 10
    
    
    if(verbose_ss is None):
        verbose_ss = False
    
    if(verbose_cter is None):
        verbose_cter = False
        
    if(verbose_freq is None):
        verbose_freq = 1
    
    # Initiating parameters for time integration 
    dt_max = (tend - tini) / (nt-1)
    t = tini
    dt = dt_max
    # params.sim_time = t*params.t_c
    
    yini = np.atleast_1d(yini)
    
    y = []
    time_t = []
    y.append(yini)
    time_t.append(t) 
    
    # Sparsity pattern 
    uband=6; lband=-uband
    offsets = [i for i in range(lband,uband)]
    ###################################################################################
    
    # Electrolyte subsystem
    ne = options_electrolyte['nCells']
    sparsity_pattern_e = sparse.diags(diagonals=[np.ones((2*ne+4 - abs(i))) for i in offsets],
                                      offsets=offsets)    
    mass_e = np.diag(options_electrolyte['mass'])
    
    def residual_e(uip1, uip, tip1, dt, cs, phis):
        return np.dot(mass_e, (uip1 - uip)) - dt*np.array(fcn_e(tip1, uip1, cs, phis))
    #####################################################################################
    
    # Solid subsystem
    ns = options_cathode['nCells']
    sparsity_pattern_s = sparse.diags(diagonals=[np.ones((2*ns+2 - abs(i))) for i in offsets],
                                      offsets=offsets)    
    
    mass_s = np.diag(options_cathode['mass'])
    
    def residual_s(uip1, uip, tip1, dt, ce, phie):
        return np.dot(mass_s, (uip1 - uip)) - dt*np.array(fcn_s(tip1, uip1, ce, phie))
    #####################################################################################
    
    def check_consistency(var_e, coupling_e, var_s, coupling_s, tol):
        cond_ce = np.abs(var_e[0] - coupling_e[0])/(tol + np.abs(coupling_e[0])*tol) 
        cond_phie = np.abs(var_e[1] - coupling_e[1])/(tol + np.abs(coupling_e[1])*tol)
        cond_cs = np.abs(var_s[0] - coupling_s[0])/(tol + np.abs(coupling_s[0])*tol)
        cond_phis = np.abs(var_s[1] - coupling_s[1])/(tol + np.abs(coupling_s[1])*tol)
        return (max(cond_ce, cond_phie, cond_cs, cond_phis) <= 1.)
        # return (cond_ce <= tol and cond_phie <= tol and cond_cs <= tol and cond_phis <= tol)
    #####################################################################################
    
    
    # Time loop starts
    print("---------------------------------- \nTime integration begins")
    while(t < tend):
        dt = min(4*dt, dt_max, tend-t)
        yn = y[-1]
        yn_e = yn[:2*ne+4]    
        yn_s = yn[2*ne+4:]
        
        ynp1_e = yn[:2*ne+4] 
        ynp1_s = yn[2*ne+4:]
        
        c_itr = 0
        # Consistency loop
        while (1):
            
            ###################
            # Solve electrolyte
            ynp1_coupling_s = ynp1_s[:2]    
            ynp1_coupling_e = ynp1_e[-2:]
            
            solx_e = newton.root_solve(lambda x: residual_e(uip1=x,
                                                            uip=yn_e, 
                                                            tip1=t+dt,
                                                            dt=dt,
                                                            cs=ynp1_coupling_s[0],
                                                            phis=ynp1_coupling_s[1]),
                                                            ynp1_e, verbose_ss,
                                                            sparsity_pattern_e,
                                                            rtol=sub_system_rtol,
                                                            atol=sub_system_rtol/10.) 
            ynp1_e = solx_e
            ###################################################################
            
            # Solve solid
            solx_s = newton.root_solve(lambda x: residual_s(uip1=x,
                                                            uip=yn_s, 
                                                            tip1=t+dt,
                                                            dt=dt,
                                                            ce=ynp1_coupling_e[0],
                                                            phie=ynp1_coupling_e[1]),
                                                            ynp1_s, verbose_ss,
                                                            sparsity_pattern_s,
                                                            rtol=sub_system_rtol,
                                                            atol=sub_system_rtol/10.)
            ynp1_s = solx_s
            ###################################################################
           
            # Check consistency
            c_itr += 1
            
            if (check_consistency(ynp1_e[-2:], ynp1_coupling_e, ynp1_s[:2], ynp1_coupling_s, c_tol)):
                if(verbose_cter):
                    print(f"At time = {(t+dt)*options_electrolyte['parameters']['t_c']:.4f} s, consistency satisfied after {c_itr} iterations")
                break
                
            if(c_itr>=max_c_itr):
                if(verbose_cter):
                    print(f"At time = {(t+dt)*options_electrolyte['parameters']['t_c']:.4f} s, max iterations reached")
                dt = dt/4
                assert dt>1e-12, "time step is too low"
                c_itr = 0
                continue
        
        y.append(np.r_[ynp1_e, ynp1_s])
        t = t + dt
        time_t.append(t)
        # params.sim_time = t*options_electrolyte['parameters']['t_c']
        
        # if (params.sim_time >= 0.4*params.tend_dim and params.sim_time < 0.5*params.tend_dim):
        #     params.xi = 0.0
        # elif (params.sim_time >= 0.5*params.tend_dim): 
        #     params.xi = -0.2
                
        if(verbose_cter):
            if(len(time_t)%verbose_freq==0):
                print(f"t = {time_t[-1]*options_electrolyte['parameters']['t_c']:.2f} s \t \t V = {y[-1][-1]*options_electrolyte['parameters']['phi_c']:.8f} volts")      
                
    if(verbose_cter):
        print(f"t = {time_t[-1]*options_electrolyte['parameters']['t_c']:.2f} s \t \t V = {y[-1][-1]*options_electrolyte['parameters']['phi_c']:.8f} volts")
        print("Simulation completed")
        print("--------------------------------------------------------------------------------")
    else:
        print(".\n.\nSimulation completed")
        print(f"time = {time_t[-1]*options_electrolyte['parameters']['t_c']:.2f} s \t voltage = {y[-1][-1]*options_electrolyte['parameters']['phi_c']:.8f} volts")
        print("--------------------------------------------------------------------------------")   
        
    return dae_result(np.array(y).T, np.array(time_t))