#!/usr/bin/env python3
# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:23:17 2022

@author: ali.asad
"""
import numpy as np
from src.lib.model import bv_current as bv
####################################################################################################################################################

# zeta parameters

def zeta_emcA(D, options_electrolyte):
    inv_N_e_c = options_electrolyte['parameters']['inv_N_e_c']
    t_0_plus = options_electrolyte['parameters']['t_0_plus']
    F = options_electrolyte['parameters']['F']
    
    return (inv_N_e_c/F)*((1.0-t_0_plus)/D)
    
def zeta_eccA(c, D, kappa, options_electrolyte):
    inv_i_e_c = options_electrolyte['parameters']['inv_i_e_c']
    beta = options_electrolyte['parameters']['beta']
    
    return inv_i_e_c*(1.0/kappa)*(1.0 + beta*kappa/(D*c))
    
def zeta_emcC(D, options_electrolyte):
    inv_N_e_c = options_electrolyte['parameters']['inv_N_e_c']
    t_0_plus = options_electrolyte['parameters']['t_0_plus']
    F = options_electrolyte['parameters']['F']
    
    return (inv_N_e_c/F)*((1.0-t_0_plus)/D)
    
def zeta_eccC(c, D, kappa, options_electrolyte):
    inv_i_e_c = options_electrolyte['parameters']['inv_i_e_c']
    beta = options_electrolyte['parameters']['beta']
    
    return inv_i_e_c*(1.0/kappa)*(1.0 + beta*kappa/(D*c))
    
def zeta_smcC(D, options_electrolyte, options_cathode):
    inv_N_s_c = options_cathode['parameters']['inv_N_s_c']
    F = options_electrolyte['parameters']['F']
    
    return (inv_N_s_c/F)*(1.0/D)
    
def zeta_sccC(sigma, options_cathode):
    inv_i_s_c = options_cathode['parameters']['inv_i_s_c'] 
    return inv_i_s_c/sigma
####################################################################################################################################################
    
# Auxiliary functions 
def G_aux_a(c1, phi1, v_aux_a, options_electrolyte):
    dx = options_electrolyte['mesh']['cellSize'][0]
    D_e = options_electrolyte['parameters']['D_e']
    c_e_c = options_electrolyte['parameters']['c_e_c']
    phi_c = options_electrolyte['parameters']['phi_c']
    kappa_e = options_electrolyte['parameters']['kappa_e']
    
    g_aux_emc = 2.0*(c1-v_aux_a[0])/dx + zeta_emcA(D_e, options_electrolyte)*bv.ise_A(v_aux_a[0]*c_e_c, v_aux_a[1]*phi_c, options_electrolyte) 
    g_aux_ecc = 2.0*(phi1-v_aux_a[1])/dx + zeta_eccA(v_aux_a[0], kappa_e, D_e, options_electrolyte)*bv.ise_A(v_aux_a[0]*c_e_c, v_aux_a[1]*phi_c, options_electrolyte)
    return np.r_[g_aux_emc, g_aux_ecc]

def G_aux_c_e(t, c, phi, v_aux_c, options_electrolyte, options_cathode, alg=None):
    if alg is None:
        alg = False
    
    dx = options_electrolyte['mesh']['cellSize'][-1]
    D_e = options_electrolyte['parameters']['D_e']
    c_e_c = options_electrolyte['parameters']['c_e_c']
    phi_c = options_electrolyte['parameters']['phi_c']
    c_s_c = options_cathode['activematerial']['parameters']['c_s_c']
    kappa_e = options_electrolyte['parameters']['kappa_e']
    
    if alg or options_electrolyte['sim_type'] in ["monolithic", "md_coupling_vars"]:  
        ise_C = bv.ise_C(v_aux_c[0]*c_e_c, v_aux_c[1]*phi_c, v_aux_c[2]*c_s_c, v_aux_c[3]*phi_c, options_electrolyte, options_cathode)
    elif options_electrolyte['sim_type']=="md_sim_flux":
        ise_C = options_electrolyte['coupling_flux'](t)
    else:
        raise Exception ('Define simulation type in electrolyte')
    
    g_aux_emc = 2.0*(v_aux_c[0]-c)/dx - zeta_emcC(D_e, options_electrolyte)*ise_C 
    g_aux_ecc = 2.0*(v_aux_c[1]-phi)/dx - zeta_eccC(v_aux_c[0], kappa_e, D_e, options_electrolyte)*ise_C
    return np.r_[g_aux_emc, g_aux_ecc]

def G_aux_c_s(t, c, phi, v_aux_c, options_electrolyte, options_cathode, alg=None):
    if alg is None:
        alg = False
    
    dx = options_cathode['activematerial']['mesh']['cellSize'][0]
    c_e_c = options_electrolyte['parameters']['c_e_c']
    phi_c = options_electrolyte['parameters']['phi_c']
    c_s_c = options_cathode['activematerial']['parameters']['c_s_c']
    D_am = options_cathode['activematerial']['parameters']['D_am']
    sigma_am = options_cathode['activematerial']['parameters']['sigma_am']
    
    if alg or options_cathode['sim_type'] in ["monolithic", "md_coupling_vars"]:  
        ise_C = bv.ise_C(v_aux_c[0]*c_e_c, v_aux_c[1]*phi_c, v_aux_c[2]*c_s_c, v_aux_c[3]*phi_c, options_electrolyte, options_cathode)
    elif options_cathode['sim_type']=="md_sim_flux":
        ise_C = options_cathode['coupling_flux'](t)
        phi = options_cathode['coupling_var_phi_s'](t)
    else:
        raise Exception ('Define simulation type in solid')
    
    g_aux_smc = 2.0*(c-v_aux_c[2])/dx - zeta_smcC(D_am, options_electrolyte, options_cathode)*ise_C
    g_aux_scc = 2.0*(phi-v_aux_c[3])/dx - zeta_sccC(sigma_am, options_cathode)*ise_C
    return np.r_[g_aux_smc, g_aux_scc]

def G_aux_c(t, ce, phie, cs, phis, v_aux_c, options_electrolyte, options_cathode, alg=None):
    if alg is None:
        alg = False
    Gg_aux_c_e = G_aux_c_e(t, ce, phie, v_aux_c, options_electrolyte, options_cathode, alg=alg)
    Gg_aux_c_s = G_aux_c_s(t, cs, phis, v_aux_c, options_electrolyte, options_cathode, alg=alg)
    return np.r_[Gg_aux_c_e, Gg_aux_c_s]

####################################################################################################################################################
####################################################################################################################################################
