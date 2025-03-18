#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:14:52 2022

@author: ali.asad
"""
import numpy as np
from src.lib.model import ocp_graphite as Uocp

####################################################################################################################################################
# Butler-Volmer current

# BV current at Anode
def ise_A(c_e_plus, phi_e_plus, options_electrolyte):
    F = options_electrolyte['parameters']['F']
    R = options_electrolyte['parameters']['R']
    T = options_electrolyte['parameters']['T']
    
    i0 = options_electrolyte['parameters']['i_0_Li']
    eta = -phi_e_plus
    
    # csMax = 0.02639*1e6
    # k0 = 0.0316227766
    # i0 = k0*np.sqrt(c_e_plus*1e3)
    
    
    ise = 2.0*i0*np.sinh(F*eta/(2.0*R*T))
    return ise

# BV current at Cathode
def ise_C(c_e_minus, phi_e_minus, c_s_plus, phi_s_plus, options_electrolyte, options_cathode):
    F = options_electrolyte['parameters']['F']
    R = options_electrolyte['parameters']['R']
    T = options_electrolyte['parameters']['T']
    k_0_C = options_electrolyte['parameters']['k_0_C']
    c_s_max = options_cathode['activematerial']['parameters']['c_s_max']
    alpha = 0.5
    
    U0 = Uocp.Ueq(c_s_plus/c_s_max)
    eta = phi_s_plus-phi_e_minus-U0
    i0 = F*k_0_C*(c_e_minus**(alpha))*(c_s_plus**(1.0-alpha))*((c_s_max-c_s_plus)**(alpha))
    
    ise = 2.0*i0*np.sinh(F*eta/(2.0*R*T)) 
    return ise
####################################################################################################################################################
