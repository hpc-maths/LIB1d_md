#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:42:54 2022

@author: ali.asad
"""
import numpy as np
from src.lib.model import fluxes as flx
from src.lib.model import fluxes_v2 as flx2
from src.lib.model import aux_system as aux_sys
from src.lib.jac_reorder import  jac_reorder as tf
####################################################################################################################################################

# Definition of f_e(u_e, t, c_s_0_plus, phi_s_0_plus) : System in Electrolyte
def func_U_e(t, u, cs=None, phis=None, options_electrolyte=None, options_cathode=None):
    if cs is None and options_electrolyte['sim_type']=="monolithic":
        raise Exception('coupling variable, cs is missing in electrolyte')
    if phis is None and options_electrolyte['sim_type']=="monolithic":
        raise Exception('coupling variable, phis is missing in electrolyte')
    
    if options_electrolyte is None:
        raise Exception('options for cathode not passed in electrolyte')
    if options_cathode is None:
        raise Exception('options for cathode not passed in electrolyte')
        
    u_aux_a, u_c_e, u_phi_e, u_aux_c_e = tf.rev_X_e(u, options_electrolyte['nCells'])  
    
    if (options_electrolyte['sim_type']=="md_coupling_vars"):
        u_aux_c_s = options_cathode['coupling_vars'](t)
    elif(options_electrolyte['sim_type']=="cosim_flux"):
        u_aux_c_s = np.r_[cs, phis] # Pass None vales to flx_Ne, flx_ie and G_aux_c_e
    elif(options_electrolyte['sim_type']=="monolithic"):
        u_aux_c_s = np.r_[cs, phis]
    else:
        raise Exception ('Define simulation type in electrolyte')    
    
    u_aux_c = np.r_[u_aux_c_e, u_aux_c_s]
    
    dx_e = options_electrolyte['mesh']['cellSize']
    
    Fe = flx.flx_Ne_f(t, u_c_e, u_phi_e, u_aux_a, u_aux_c, options_electrolyte, options_cathode)
    Ge = flx.flx_ie_f(t, u_c_e, u_phi_e, u_aux_a, u_aux_c, options_electrolyte, options_cathode)
    
    Ffe = (Fe[1:] - Fe[0:-1])/dx_e[:]    
    Gge = (Ge[1:] - Ge[0:-1])/dx_e[:]
    
    Gg_aux_a = aux_sys.G_aux_a(u_c_e[0], u_phi_e[0],  u_aux_a, options_electrolyte)
    Gg_aux_c_e = aux_sys.G_aux_c_e(t, u_c_e[-1], u_phi_e[-1], u_aux_c, options_electrolyte, options_cathode)
    
    return np.r_[Gg_aux_a, tf.gather_Xi(Ffe, Gge), Gg_aux_c_e]
####################################################################################################################################################

# Definition of f_e(u_e, t, c_s_0_plus, phi_s_0_plus) : System in Solid
def func_U_s(t, u, ce=None, phie=None, options_electrolyte=None, options_cathode=None):
    
    if ce is None and options_cathode['sim_type']=="monolithic":
        raise Exception('coupling variable, ce is missing in solid')
    if phie is None and options_cathode['sim_type']=="monolithic":
        raise Exception('coupling variable, phie is missing in solid')
        
    if options_electrolyte is None:
        raise Exception('options for cathode not passed in solid')
    if options_cathode is None:
        raise Exception('options for cathode not passed in solid')
        
    u_aux_c_s, u_c_s, u_phi_s = tf.rev_X_s(u, options_cathode['nCells'])        
    
    if (options_cathode['sim_type']=="md_coupling_vars"):
        u_aux_c_e = options_electrolyte['coupling_vars'](t)
    elif(options_cathode['sim_type']=="cosim_flux"):
        u_aux_c_e = np.r_[ce, phie] # Pass None vales to flx_Ns, flx_is and G_aux_c_s
    elif(options_cathode['sim_type']=="monolithic"):
        u_aux_c_e = np.r_[ce, phie]
    else:
        raise Exception ('Define simulation type in solid')
        
    u_aux_c = np.r_[u_aux_c_e, u_aux_c_s]
        
    dx_s = np.r_[options_cathode['activematerial']['mesh']['cellSize'], options_cathode['currentcollector']['mesh']['cellSize']]
        
    Fs = flx.flx_Ns_f(t, u_c_s, u_aux_c, options_electrolyte, options_cathode)
    Gs = flx.flx_is_f(t, u_phi_s, u_aux_c, options_electrolyte, options_cathode)
    
    varepsln = options_cathode['parameters']['varepsln']
    
    Ffs = varepsln*(Fs[1:] - Fs[0:-1])/dx_s[:]
    Ggs = (Gs[1:] - Gs[0:-1])/dx_s[:]
    
    if (options_cathode['sim_type']=="cosim_flux"):
        sigma_am = options_cathode['activematerial']['parameters']['sigma_am']
        Ggs[0] = (Gs[1] - 2.0*sigma_am*(u_phi_s[0] - options_cathode['coupling_var_phi_s'](t))/dx_s[0])/dx_s[0]
    
    Gg_aux_c_s = aux_sys.G_aux_c_s(t, u_c_s[0], u_phi_s[0], u_aux_c, options_electrolyte, options_cathode)
    
    return np.r_[Gg_aux_c_s, tf.gather_Xi(Ffs, Ggs)]
####################################################################################################################################################

# Definition of full system combining electrolyte and solid
# Only called for a fully coupled monolithic simulation
def func_U(u, t, options_electrolyte, options_cathode):
    ne = options_electrolyte['nCells']
    
    func_e = func_U_e(t, u[:2*ne+4], u[2*ne+4], u[2*ne+5], options_electrolyte, options_cathode)
    func_s = func_U_s(t, u[2*ne+4:], u[2*ne+2], u[2*ne+3], options_electrolyte, options_cathode)   
    return np.r_[func_e, func_s]

####################################################################################################################################################
####################################################################################################################################################
# Algebraic system
# Defined in a coupled fashion as it is only used
# in the begining of the simulation for a good initial guess
def func_U_alg(u, options_electrolyte, options_cathode):
    ne = options_electrolyte['nCells']
    ns = options_cathode['nCells']
    t = 0.0
    
    dx_e = options_electrolyte['mesh']['cellSize']
    dx_s = np.r_[options_cathode['activematerial']['mesh']['cellSize'], options_cathode['currentcollector']['mesh']['cellSize']] 
    
    u_c_e = np.ones(ne)
    u_c_s = (options_cathode['activematerial']['parameters']['c_s_i']/options_cathode['activematerial']['parameters']['c_s_max'])*np.ones(ns)
    
    u_aux_a = u[:2]
    u_phi_e = u[2:ne+2]
    u_aux_c = u[ne+2:ne+6]
    u_phi_s = u[ne+6:]      
    
        
    # Ge = flx2.flx_ie_f(t, u_c_e, u_phi_e, u_aux_a, u_aux_c, options_electrolyte, options_cathode, alg=True)
    # Gs = flx2.flx_is_f(t, u_phi_s, u_aux_c, options_electrolyte, options_cathode, alg=True)
          
    Ge = flx.flx_ie_f(t, u_c_e, u_phi_e, u_aux_a, u_aux_c, options_electrolyte, options_cathode, alg=True)
    Gs = flx.flx_is_f(t, u_phi_s, u_aux_c, options_electrolyte, options_cathode, alg=True)
    
    
    Gge = (Ge[1:] - Ge[0:-1])/dx_e[:]
    Ggs = (Gs[1:] - Gs[0:-1])/dx_s[:]
    
    Gg_aux_a = aux_sys.G_aux_a(u_c_e[0], u_phi_e[0],  u_aux_a, options_electrolyte)
    Gg_aux_c = aux_sys.G_aux_c(t, u_c_e[-1], u_phi_e[-1], u_c_s[0], u_phi_s[0], u_aux_c, options_electrolyte, options_cathode, alg=True)
    
    return np.r_[Gg_aux_a, Gge, Gg_aux_c, Ggs]
####################################################################################################################################################
###############################################################
#####################################################################################

# New 1D model based on fluxes that do not use i_BV inside domains 
#%%


def func_Ue2(t, u, cs=None, phis=None, options_electrolyte=None, options_cathode=None):
    if cs is None and options_electrolyte['sim_type']=="monolithic":
        raise Exception('coupling variable, cs is missing in electrolyte')
    if phis is None and options_electrolyte['sim_type']=="monolithic":
        raise Exception('coupling variable, phis is missing in electrolyte')
    
    if options_electrolyte is None:
        raise Exception('options for cathode not passed in electrolyte')
    if options_cathode is None:
        raise Exception('options for cathode not passed in electrolyte')
        
    u_aux_a, u_c_e, u_phi_e, u_aux_c_e = tf.rev_X_e(u, options_electrolyte['nCells'])  
    
    if (options_electrolyte['sim_type']=="md_coupling_vars"):
        u_aux_c_s = options_cathode['coupling_vars'](t)
    elif(options_electrolyte['sim_type']=="cosim_flux"):
        u_aux_c_s = np.r_[cs, phis] # Pass None vales to flx_Ne, flx_ie and G_aux_c_e
    elif(options_electrolyte['sim_type']=="monolithic"):
        u_aux_c_s = np.r_[cs, phis]
    else:
        raise Exception ('Define simulation type in electrolyte')    
    
    u_aux_c = np.r_[u_aux_c_e, u_aux_c_s]
    
    dx_e = options_electrolyte['mesh']['cellSize']
    
    Fe = flx2.flx_Ne_f(t, u_c_e, u_phi_e, u_aux_a, u_aux_c, options_electrolyte, options_cathode)
    Ge = flx2.flx_ie_f(t, u_c_e, u_phi_e, u_aux_a, u_aux_c, options_electrolyte, options_cathode)
    
    Ffe = (Fe[1:] - Fe[0:-1])/dx_e[:]    
    Gge = (Ge[1:] - Ge[0:-1])/dx_e[:]
    
    Gg_aux_a = aux_sys.G_aux_a(u_c_e[0], u_phi_e[0],  u_aux_a, options_electrolyte)
    Gg_aux_c_e = aux_sys.G_aux_c_e(t, u_c_e[-1], u_phi_e[-1], u_aux_c, options_electrolyte, options_cathode)
    
    return np.r_[Gg_aux_a, tf.gather_Xi(Ffe, Gge), Gg_aux_c_e]
####################################################################################################################################################

# Definition of f_e(u_e, t, c_s_0_plus, phi_s_0_plus) : System in Solid
def func_Us2(t, u, ce=None, phie=None, options_electrolyte=None, options_cathode=None):
    
    if ce is None and options_cathode['sim_type']=="monolithic":
        raise Exception('coupling variable, ce is missing in solid')
    if phie is None and options_cathode['sim_type']=="monolithic":
        raise Exception('coupling variable, phie is missing in solid')
        
    if options_electrolyte is None:
        raise Exception('options for cathode not passed in solid')
    if options_cathode is None:
        raise Exception('options for cathode not passed in solid')
        
    u_aux_c_s, u_c_s, u_phi_s = tf.rev_X_s(u, options_cathode['nCells'])        
    
    if (options_cathode['sim_type']=="md_coupling_vars"):
        u_aux_c_e = options_electrolyte['coupling_vars'](t)
    elif(options_cathode['sim_type']=="cosim_flux"):
        u_aux_c_e = np.r_[ce, phie] # Pass None vales to flx_Ns, flx_is and G_aux_c_s
    elif(options_cathode['sim_type']=="monolithic"):
        u_aux_c_e = np.r_[ce, phie]
    else:
        raise Exception ('Define simulation type in solid')
        
    u_aux_c = np.r_[u_aux_c_e, u_aux_c_s]
        
    dx_s = np.r_[options_cathode['activematerial']['mesh']['cellSize'], options_cathode['currentcollector']['mesh']['cellSize']]
        
    Fs = flx2.flx_Ns_f(t, u_c_s, u_aux_c, options_electrolyte, options_cathode)
    Gs = flx2.flx_is_f(t, u_phi_s, u_aux_c, options_electrolyte, options_cathode)
    
    varepsln = options_cathode['parameters']['varepsln']
    
    Ffs = varepsln*(Fs[1:] - Fs[0:-1])/dx_s[:]
    Ggs = (Gs[1:] - Gs[0:-1])/dx_s[:]
    
    if (options_cathode['sim_type']=="cosim_flux"):
        sigma_am = options_cathode['activematerial']['parameters']['sigma_am']
        Ggs[0] = (Gs[1] - 2.0*sigma_am*(u_phi_s[0] - options_cathode['coupling_var_phi_s'](t))/dx_s[0])/dx_s[0]
    
    Gg_aux_c_s = aux_sys.G_aux_c_s(t, u_c_s[0], u_phi_s[0], u_aux_c, options_electrolyte, options_cathode)
    
    return np.r_[Gg_aux_c_s, tf.gather_Xi(Ffs, Ggs)]
####################################################################################################################################################

# Definition of full system combining electrolyte and solid
# Only called for a fully coupled monolithic simulation
def func_U2(u, t, options_electrolyte, options_cathode):
    ne = options_electrolyte['nCells']
    
    func_e = func_Ue2(t, u[:2*ne+4], u[2*ne+4], u[2*ne+5], options_electrolyte, options_cathode)
    func_s = func_Us2(t, u[2*ne+4:], u[2*ne+2], u[2*ne+3], options_electrolyte, options_cathode)   
    return np.r_[func_e, func_s]


#%%




