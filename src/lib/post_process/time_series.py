# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:55:09 2022

@author: ali.asad
"""

import numpy as np
from src.lib.jac_reorder import  jac_reorder as tf
####################################################################################################################################################

# Interpret solutions
def get_y_t(t, y, options_electrolyte, options_cathode, bNonDim=False):
    num_t = t.size
    time_t = t
    L_c = options_electrolyte['parameters']['L_c']
    x = (np.r_[options_electrolyte['mesh']['cellX'], options_cathode['activematerial']['mesh']['cellX'], options_cathode['currentcollector']['mesh']['cellX']])
    
    y_c_e_t = np.zeros([options_electrolyte['nCells'], num_t])
    y_phi_e_t = np.zeros([options_electrolyte['nCells'], num_t])
    y_c_s_t = np.zeros([options_cathode['nCells'], num_t])
    y_phi_s_t = np.zeros([options_cathode['nCells'], num_t])
    y_aux_a_t = np.zeros([2, num_t])
    y_aux_c_t = np.zeros([4, num_t])

    for i in range(num_t):
        y_aux_a_t[:, i], y_c_e_t[:, i], y_phi_e_t[:, i], y_aux_c_t[:, i], y_c_s_t[:, i], y_phi_s_t[:, i] = tf.rev_X(y[:, i],
                                                                                                                    options_electrolyte['nCells'],
                                                                                                                    options_cathode['nCells'])
    if not bNonDim:
        # making variables dimensional
        time_tt = time_t * options_electrolyte['parameters']['t_c']
        x *= L_c
        y_c_e_t *= options_electrolyte['parameters']['c_e_c']
        y_c_s_t *= options_cathode['activematerial']['parameters']['c_s_c']
        y_phi_e_t *= options_electrolyte['parameters']['phi_c']
        y_phi_s_t *= options_cathode['parameters']['phi_c']
        y_aux_a_t[0] *= options_electrolyte['parameters']['c_e_c']
        y_aux_a_t[1] *= options_electrolyte['parameters']['phi_c']
        y_aux_c_t[0] *= options_electrolyte['parameters']['c_e_c']
        y_aux_c_t[1] *= options_electrolyte['parameters']['phi_c']
        y_aux_c_t[2] *= options_cathode['activematerial']['parameters']['c_s_c']
        y_aux_c_t[3] *= options_cathode['parameters']['phi_c']
    else:
        time_tt = time_t
        
    return x, time_tt, y_aux_a_t, y_c_e_t, y_phi_e_t, y_aux_c_t, y_c_s_t, y_phi_s_t

def get_diff_t(t, y, options_electrolyte, options_cathode, bNonDim=False):
    num_t = t.size
    time_t = t
    x = (np.r_[options_electrolyte['mesh']['cellX'], options_cathode['activematerial']['mesh']['cellX'], options_cathode['currentcollector']['mesh']['cellX']])
    
    y_c_e_t = np.zeros([options_electrolyte['nCells'], num_t])
    y_phi_e_t = np.zeros([options_electrolyte['nCells'], num_t])
    y_c_s_t = np.zeros([options_cathode['nCells'], num_t])
    y_phi_s_t = np.zeros([options_cathode['nCells'], num_t])
    y_aux_a_t = np.zeros([2, num_t])
    y_aux_c_t = np.zeros([4, num_t])

    for i in range(num_t):
        y_aux_a_t[:, i], y_c_e_t[:, i], y_phi_e_t[:, i], y_aux_c_t[:, i], y_c_s_t[:, i], y_phi_s_t[:, i] = tf.rev_X(y[:, i],
                                                                                                                    options_electrolyte['nCells'],
                                                                                                                    options_cathode['nCells'])
    if not bNonDim:
        # making variables dimensional
        x_out = x*options_electrolyte['parameters']['L_c']
        t_out = time_t*options_electrolyte['parameters']['t_c']
        ce_t_out = y_c_e_t*options_electrolyte['parameters']['c_e_c']
        cs_t_out = y_c_s_t*options_cathode['activematerial']['parameters']['c_s_c']
    else:
        x_out = x
        t_out = time_t
        ce_t_out = y_c_e_t
        cs_t_out = y_c_s_t
        
    return x_out, t_out, ce_t_out, cs_t_out