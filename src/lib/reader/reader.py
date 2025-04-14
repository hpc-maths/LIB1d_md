# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:45:42 2022

@author: ali.asad
"""

import json
################################################################################################################################################### 

def readInputs(input_dir):
    if (input_dir[-1] != "/"):
        input_dir = input_dir + "/"

    # Loading physical parameters JSON file
    fname_phy = input_dir + "params_phy.json"

    f_param = open(fname_phy)
    data_phy = json.load(f_param)
    
    fname_sim = input_dir + "params_num.json"

    # Loading numerical simulation parameters JSON file
    f_sim = open(fname_sim)
    data_sim = json.load(f_sim)
    
    return data_phy[0], data_sim[0]
###################################################################################################################################################
    
# def globalConstantParameters(dict_phy, dict_sim):
# # Constants
#     global F, R, T
#     F = dict_phy['Constants']['F']['value']
#     R = dict_phy['Constants']['R']['value']
#     T = dict_phy['Constants']['T']['value']
# ###################################################################################################################################################

# # Electrolyte
#     global L_e, c_e_i, D_e_c, kappa_e_c, t_0_plus, del_e
#     L_e = dict_phy['Electrolyte']['L_e']['value']
#     c_e_i = dict_phy['Electrolyte']['c_e_i']['value']
#     D_e_c = dict_phy['Electrolyte']['D_e_c']['value']
#     kappa_e_c = dict_phy['Electrolyte']['kappa_e_c']['value']
#     t_0_plus = dict_phy['Electrolyte']['t_0_plus']['value']
#     del_e = dict_phy['Electrolyte']['del_e']['value']
# ###################################################################################################################################################

# # Electrode
#     global L_am, c_s_max, D_am_c, sigma_am_c
#     L_am = dict_phy['Electrode']['L_am']['value']
#     c_s_max = dict_phy['Electrode']['c_s_max']['value']
#     D_am_c = dict_phy['Electrode']['D_am_c']['value']
#     sigma_am_c = dict_phy['Electrode']['sigma_am_c']['value']
# ###################################################################################################################################################

# # Current Collector
#     global L_cc, sigma_cc_c
#     L_cc = dict_phy['CurrentCollector']['L_cc']['value']
#     sigma_cc_c = dict_phy['CurrentCollector']['sigma_cc_c']['value']
# ###################################################################################################################################################

# # Butler-Volmer parameters
#     global i_0_Li, k_0_C
#     i_0_Li = dict_phy['ButlerVolmerParams']['i_0_Li']['value']
#     k_0_C = dict_phy['ButlerVolmerParams']['k_0_C']['value']
#     k_0_C = k_0_C/F
# ###################################################################################################################################################

# # Mesh parameters
#     global ne, nam, ncc, ns, n_all
#     ne = dict_sim['Mesh']['ne']['value']
#     nam = dict_sim['Mesh']['nam']['value']
#     ncc = dict_sim['Mesh']['ncc']['value']
#     ns = nam + ncc
#     n_all = ne + ns
# ###################################################################################################################################################

# # Time integration parameters
#     global tend_dim, tini_dim, dt_dim, sim_time
#     tend_dim = dict_sim['TimeIntegration']['tend']['value']
#     tini_dim = dict_sim['TimeIntegration']['tini']['value']
#     dt_dim = dict_sim['TimeIntegration']['dt']['value']
#     sim_time = tini_dim
# ###################################################################################################################################################

# # Length and time scales
#     global L_s, L_c, t_c
#     L_s = L_am + L_cc
#     L_c = L_e + L_am + L_cc
#     t_c = (L_c**2.0)/D_e_c


# ###############################################################################
# # Charge Rate and External circuit current
#     global xi, i_1C
#     xi = dict_sim['ChargeRate']['xi']['value']

#     i_1C = F*c_s_max*L_am/3600.0
# ###############################################################################

# # Non-dimensional initial variables

# # Electrolyte
#     global c_e_c, phi_e_i, c_s_c, c_s_i, phi_s_i, phi_c
#     c_e_c = c_e_i
#     c_e_i = c_e_i/c_e_c
#     phi_e_i = 0.0

# ##Active material and current collector
#     c_s_c = c_s_max
#     if (xi > 0.0):
#         c_s_i = 13000.0/c_s_c
#         print("--------------------------------------------------------------------------------")
#         print(f"CHARGING at charge rate = {xi:.2f} C\n----------------------------------")
#     else:
#         c_s_i = 1000.0/c_s_c
#         print("--------------------------------------------------------------------------------")
#         print(f"DISCHARGING at charge rate = {xi:.2f} C\n----------------------------------")
#     phi_s_i = 0.0

#     phi_c = R*T/F
# ###############################################################################    

# # Non-dimensional parameters for the model
#     global D_e, kappa_e, D_am, sigma_c, sigma_am, sigma_cc
# ## Electrolyte
#     D_e = D_e_c/D_e_c
#     kappa_e = kappa_e_c/kappa_e_c

# ## Active material
#     D_am = D_am_c/D_am_c 
#     sigma_c = sigma_am_c
#     sigma_am = sigma_am_c/sigma_c 

# ## Current collector
#     sigma_cc = sigma_cc_c/sigma_c
# ###############################################################################

# # Grid
#     global x, dx
#     x = np.linspace(0, L_c, num=n_all+1)/L_c
#     dx = x[1]-x[0]
#     for i in range(x.size-1):
#         x[i] = x[i+1]+x[i]
#     x = x[:-1]/2.0
# ###############################################################################

# # Inverse flux scales
#     global inv_i_e_c, inv_i_s_c, inv_N_e_c, inv_N_s_c
#     inv_i_e_c = (L_c/(phi_c*kappa_e_c))
#     inv_i_s_c = (L_c/(phi_c*sigma_c))
#     inv_N_e_c = (L_c/(c_e_c*D_e_c))
#     inv_N_s_c = (L_c/(c_s_c*D_am_c))
# ###############################################################################

# # Problem parmeters
#     global Lambda, beta, varepsln
#     Lambda = 2*R*T*(1.0-t_0_plus)*(1+del_e)/F
#     beta = Lambda*kappa_e_c*(1.0-t_0_plus)/(D_e_c*c_e_c*F)
#     varepsln = D_am_c/D_e_c
# ###############################################################################

# # Simulation (solver) parameters
#     global tend, tini, nt
#     tend = tend_dim/t_c
#     tini = tini_dim/t_c
#     nt = int((tend_dim-tini_dim)/dt_dim) + 1
# ###############################################################################
