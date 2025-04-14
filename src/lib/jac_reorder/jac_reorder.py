#!/usr/bin/env python3
# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:42:29 2022

@author: ali.asad
"""
import numpy as np

# Transform functions
def gather_Xi(fi1, fi2):
    n = fi1.size
    g_xi = np.reshape([fi1, fi2], (2, n), order='C')
    return np.reshape(g_xi, (1,2*n), order='F')[0,:]

def rev_X_e(x, ne):
    aux_a_e = x[:2]
    x_e = x[2:2*ne+2]
    aux_c_e = x[2*ne+2:2*ne+4]
    m_aux_a_e = np.reshape(aux_a_e, (2, 1), order='F')
    m_xe = np.reshape(x_e, (2, ne), order='F')
    m_aux_c_e = np.reshape(aux_c_e, (2, 1), order='F')
    return np.r_[m_aux_a_e[0,:], m_aux_a_e[1,:]], m_xe[0,:], m_xe[1,:], np.r_[m_aux_c_e[0,:], m_aux_c_e[1,:]]

def rev_X_e_md_sim(x, ne):
    aux_a_e = x[:2]
    x_e = x[2:2*ne+2]
    m_aux_a_e = np.reshape(aux_a_e, (2, 1), order='F')
    m_xe = np.reshape(x_e, (2, ne), order='F')
    return np.r_[m_aux_a_e[0,:], m_aux_a_e[1,:]], m_xe[0,:], m_xe[1,:]

def rev_X_s(x, ns):
    aux_c_s = x[:2]
    x_s = x[2:2*ns+2]
    m_aux_c_s = np.reshape(aux_c_s, (2, 1), order='F')
    m_xs = np.reshape(x_s, (2, ns), order='F')
    return np.r_[m_aux_c_s[0,:], m_aux_c_s[1,:]], m_xs[0,:], m_xs[1,:]

def rev_X_s_md_sim(x, ns):
    x_s = x
    m_xs = np.reshape(x_s, (2, ns), order='F')
    return m_xs[0,:], m_xs[1,:]

def rev_X(x, ne, ns):
    aux_ae, ce, phie, aux_ce = rev_X_e(x[:2*ne+4], ne)
    aux_cs, cs, phis = rev_X_s(x[2*ne+4:], ns)
    return aux_ae, ce, phie, np.r_[aux_ce, aux_cs], cs, phis

def rev_X_md_sim(x, ne, ns):
    aux_ae, ce, phie = rev_X_e_md_sim(x[:2*ne+2], ne)
    cs, phis = rev_X_s_md_sim(x[2*ne+2:], ns)
    return aux_ae, ce, phie, cs, phis
####################################################################################################################################################
####################################################################################################################################################
