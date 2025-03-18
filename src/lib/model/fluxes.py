#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:40:04 2022

@author: ali.asad
"""
import numpy as np
from src.lib.model import bv_current as bv
####################################################################################################################################################

# Flux Definitions

# current flux density in electrolyte
def flx_ie_f(t, c, phi, v_aux_a, v_aux_c, options_electrolyte, options_cathode, alg=None):
    
    if alg is None:
        alg=False
    
    inv_i_e_c = options_electrolyte['parameters']['inv_i_e_c']
    phi_c = options_electrolyte['parameters']['phi_c']
    c_e_c = options_electrolyte['parameters']['c_e_c']
    c_s_c = options_cathode['activematerial']['parameters']['c_s_c']
    kappa_e = options_electrolyte['parameters']['kappa_e']
    Lambda = options_electrolyte['parameters']['Lambda']
    
    
    dx = options_electrolyte['mesh']['dxBetweenCellCenters']
     
    c_inv_mean = (c[1:]+c[:-1])/(2*c[1:]*c[:-1])
    flx_ie = np.zeros(c.size+1)
    ise_A = bv.ise_A(v_aux_a[0]*c_e_c, v_aux_a[1]*phi_c, options_electrolyte)
    
    if alg or options_electrolyte['sim_type'] in ["monolithic", "md_coupling_vars"]:  
        ise_C = bv.ise_C(v_aux_c[0]*c_e_c,
                         v_aux_c[1]*phi_c,
                         v_aux_c[2]*c_s_c,
                         v_aux_c[3]*phi_c,
                         options_electrolyte,
                         options_cathode)
    elif options_electrolyte['sim_type']=="cosim_flux":
        ise_C = options_electrolyte['coupling_flux'](t)
    else:
        raise Exception ('Define simulation type in electrolyte')
    
    flx_ie[0] = - inv_i_e_c*ise_A
    flx_ie[1:-1] = (kappa_e*(phi[1:]-phi[:-1])/dx[:] - (Lambda/phi_c)*kappa_e*c_inv_mean[:]*(c[1:]-c[:-1])/dx[:])
    flx_ie[-1] =  inv_i_e_c*ise_C
    
    return flx_ie
####################################################################################################################################################

# diffusive flux in electrolyte
def flx_Ne_f(t, c, phi, v_aux_a, v_aux_c, options_electrolyte, options_cathode):
    inv_N_e_c = options_electrolyte['parameters']['inv_N_e_c']
    
    phi_c = options_electrolyte['parameters']['phi_c']
    c_e_c = options_electrolyte['parameters']['c_e_c']
    c_s_c = options_cathode['activematerial']['parameters']['c_s_c']
    kappa_e_c = options_electrolyte['parameters']['kappa_e_c']
    D_e_c = options_electrolyte['parameters']['D_e_c']
    D_e = options_electrolyte['parameters']['D_e']
    F = options_electrolyte['parameters']['F']
    t_0_plus = options_electrolyte['parameters']['t_0_plus']
    
    dx = options_electrolyte['mesh']['dxBetweenCellCenters']
    
    flx_Ne = np.zeros(c.size+1)
    
    flx_ie_call = flx_ie_f(t, c, phi, v_aux_a, v_aux_c, options_electrolyte, options_cathode)
    ise_A = bv.ise_A(v_aux_a[0]*c_e_c, v_aux_a[1]*phi_c, options_electrolyte)
    
    if options_electrolyte['sim_type'] in ["monolithic", "md_coupling_vars"]:  
        ise_C = bv.ise_C(v_aux_c[0]*c_e_c,
                         v_aux_c[1]*phi_c,
                         v_aux_c[2]*c_s_c,
                         v_aux_c[3]*phi_c,
                         options_electrolyte,
                         options_cathode)
    elif options_electrolyte['sim_type']=="cosim_flux":
        ise_C = options_electrolyte['coupling_flux'](t)
    else:
        raise Exception ('Define simulation type in electrolyte')
    
    flx_Ne[0] = - inv_N_e_c*ise_A/F 
    flx_Ne[1:-1] = (D_e*(c[1:]-c[:-1])/dx[:] + (t_0_plus*kappa_e_c*phi_c/(F*D_e_c*c_e_c))*flx_ie_call[1:-1])   
    flx_Ne[-1] =  inv_N_e_c*ise_C/F
    
    return flx_Ne  
####################################################################################################################################################

# diffusive flux in solid (active material and current collector[=0])
def flx_Ns_f(t, c, v_aux_c, options_electrolyte, options_cathode):
    inv_N_s_c = options_cathode['parameters']['inv_N_s_c']
    phi_c = options_electrolyte['parameters']['phi_c']
    c_e_c = options_electrolyte['parameters']['c_e_c']
    c_s_c = options_cathode['activematerial']['parameters']['c_s_c']
    D_am = options_cathode['activematerial']['parameters']['D_am']
    F = options_electrolyte['parameters']['F']
    nam = options_cathode['activematerial']['nCells']
    dx = options_cathode['activematerial']['mesh']['dxBetweenCellCenters']
    
    flx_Ns = np.zeros(c.size + 1)
    
    if options_cathode['sim_type'] in ["monolithic", "md_coupling_vars"]:  
        ise_C = bv.ise_C(v_aux_c[0]*c_e_c,
                         v_aux_c[1]*phi_c,
                         v_aux_c[2]*c_s_c,
                         v_aux_c[3]*phi_c,
                         options_electrolyte,
                         options_cathode)
    elif options_cathode['sim_type']=="cosim_flux":
        ise_C = options_cathode['coupling_flux'](t)
    else:
        raise Exception ('Define simulation type in solid')
    
    flx_Ns[0] =  inv_N_s_c*ise_C/F 
    flx_Ns[1:nam] = D_am*(c[1:nam]-c[0:nam-1])/dx[:]
    flx_Ns[nam:] = 0.0
    
    return flx_Ns
####################################################################################################################################################

# current flux density in solid
def flx_is_f(t, phi, v_aux_c, options_electrolyte, options_cathode, alg=None):
    if alg is None:
        alg = False
        
    inv_i_s_c = options_cathode['parameters']['inv_i_s_c']
    phi_c = options_electrolyte['parameters']['phi_c']
    c_e_c = options_electrolyte['parameters']['c_e_c']
    c_s_c = options_cathode['activematerial']['parameters']['c_s_c']
    sigma_am = options_cathode['activematerial']['parameters']['sigma_am']
    sigma_cc = options_cathode['currentcollector']['parameters']['sigma_cc']
     
    nam = options_cathode['activematerial']['nCells']
    dx_am = options_cathode['activematerial']['mesh']['dxBetweenCellCenters']
    dx_cc = options_cathode['currentcollector']['mesh']['dxBetweenCellCenters']
    dx_am_cc = abs(options_cathode['activematerial']['mesh']['cellX'][-1]-options_cathode['currentcollector']['mesh']['cellX'][0])
    
    flx_is = np.zeros(phi.size + 1)
    
    if alg or options_cathode['sim_type'] in ["monolithic", "md_coupling_vars"]:  
        ise_C = bv.ise_C(v_aux_c[0]*c_e_c,
                         v_aux_c[1]*phi_c,
                         v_aux_c[2]*c_s_c,
                         v_aux_c[3]*phi_c,
                         options_electrolyte,
                         options_cathode)
    elif options_cathode['sim_type']=="cosim_flux":
        ise_C = options_cathode['coupling_flux'](t)
    else:
        raise Exception ('Define simulation type in solid')
    
    
    flx_is[0] =  inv_i_s_c*ise_C
    flx_is[1:nam] = sigma_am*(phi[1:nam]-phi[0:nam-1])/dx_am[:]
    flx_is[nam] = (2*sigma_cc*sigma_am/(sigma_cc+sigma_am))*(phi[nam]-phi[nam-1])/dx_am_cc
    flx_is[nam+1:-1] = sigma_cc*(phi[nam+1:]-phi[nam:-1])/dx_cc[:]
    
    if (options_cathode['parameters']['ChargeType'] == "CC"):
        xi = options_cathode['parameters']['xi'] 
        i_1C = options_cathode['parameters']['i_1C']
        flx_is[-1] =  inv_i_s_c*xi*i_1C
    else:
        phi_L = options_cathode['parameters']['phi_s_L'](t)
        flx_is[-1] =  2.0*sigma_cc*(phi_L/phi_c - phi[-1])/dx_cc[-1]
    
    return flx_is
####################################################################################################################################################
####################################################################################################################################################
