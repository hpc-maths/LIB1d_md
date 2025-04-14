# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:32:07 2022

@author: ali.asad
"""

import numpy as np
import matplotlib.pyplot as plt

from src.lib.post_process import time_series as y_ts
####################################################################################################################################################


def plot_cv_evol(time_t, cv_ref, cv_md_sim=None, plotType=None, compareType=None, title=" ", figsize=None, legend=True, figname=None):
    if plotType is None:
        plotType ="line"
    
    if compareType is None:
        if cv_md_sim is not None:
            compareType="difference"
            
    if plotType=="scatter":
        marker='o'
        marker2='*'
    else :
        marker=None
        marker2=None
        
    if figsize is None:
        figsize=(8,6)
    
    if figname is not None:
        savefig = True
        dpi=600
    else:
        savefig=False
        
    lwc=0.8
    lwref=1.25
    lwcos=1.0
        
    fig = plt.figure(figsize=figsize)    
    fig.suptitle(title)
    ax_l = fig.add_subplot(2,2,1)  
    
    ax_l.plot(time_t, cv_ref[3,:],
              color='r', marker=marker, lw=lwref,
              label=r"$Monolithic$")
    
    if cv_md_sim is not None:
        ax_l.plot(time_t, cv_md_sim[3,:],
                  color='b', marker=marker2, lw=lwcos, linestyle='--',
                  label=r"$md_simulation$")               
    
        ax_r = ax_l.twinx()
        
        if compareType=="difference":
            ax_r.semilogy(time_t, np.abs(cv_ref[3,:]-cv_md_sim[3,:])/cv_ref[3,:],
                      'k', lw=lwc)
            ax_l.plot(np.nan, np.nan, 'k', label=r"$Difference \ (\delta)$" )
            ax_r.set_ylabel(r"$\delta(\phi_{s, 0}^+)/\phi_{s, 0}^+ ._{ref}$")
        else:
            ax_r.plot(time_t, np.abs(cv_md_sim[3,:]/cv_ref[3,:]),
                      'k', lw=lwc)
            ax_l.plot(np.nan, np.nan, 'k', label=r"$Ratio$" )
            ax_r.set_ylabel(r"$\phi_{s, 0}^+/\phi_{s, 0}^+ ._{ref}$")
    
    
    ax_l.set_ylabel(r"$\phi_{s, 0}^+ \ (Volts)$")
    ax_l.set_xlabel(r"$t \ (s)$")
    ax_l.grid()
    if legend:
        ax_l.legend(fancybox=True)
    
    plt.tight_layout()
    ####################################################################################################################################################
        
    ax_l = fig.add_subplot(2,2,2)  
    ax_l.plot(time_t, cv_ref[2,:],
              color='r', marker=marker, lw=lwref)
    
    if cv_md_sim is not None:         
        ax_l.plot(time_t, cv_md_sim[2,:],
                  color='b', marker=marker2, lw=lwcos, linestyle='--')
        
        ax_r = ax_l.twinx()
        
        if compareType=="difference":
            ax_r.semilogy(time_t, np.abs(cv_ref[2,:]-cv_md_sim[2,:])/cv_ref[2,:],
                      'k', lw=lwc)
            ax_r.set_ylabel(r"$\delta(c_{s, 0}^+)/c_{s, 0}^+ ._{ref}$")
            
        else:
            ax_r.plot(time_t, np.abs(cv_md_sim[2,:]/cv_ref[2,:]),
                      'k', lw=lwc)
            ax_r.set_ylabel(r"$c_{s, 0}^+/c_{s, 0}^+ ._{ref}$")
        
        
    ax_l.set_ylabel(r"$c_{s, 0}^+ \ (mol/m^3)$")
    ax_l.set_xlabel(r"$t \ (s)$")
    ax_l.grid()
    plt.tight_layout()
    ####################################################################################################################################################
    
    ax_l = fig.add_subplot(2,2,3)  
    ax_l.plot(time_t, cv_ref[1,:],
              color='r', marker=marker, lw=lwref)
    
    if cv_md_sim is not None:
        ax_l.plot(time_t, cv_md_sim[1,:],
                  color='b', marker=marker2, lw=lwcos, linestyle='--')
        
        ax_r = ax_l.twinx()
        
        if compareType=="difference":
            ax_r.semilogy(time_t, np.abs(cv_ref[1,:]-cv_md_sim[1,:])/cv_ref[1,:],
                      'k', lw=lwc)
            ax_r.set_ylabel(r"$\delta(\phi_{e, -1}^-)/\phi_{e, -1}^- ._{ref}$")
        else:
            ax_r.plot(time_t, np.abs(cv_md_sim[1,:]/cv_ref[1,:]),
                      'k', lw=lwc)
            ax_r.set_ylabel(r"$\phi_{e, -1}^-/\phi_{e, -1}^- ._{ref}$")
        
    ax_l.set_ylabel(r"$\phi_{e, 0}^- \ (Volts)$")
    ax_l.set_xlabel(r"$t \ (s)$")
    ax_l.grid()
    plt.tight_layout()
    #####################################################################################################################################################
    
    ax_l = fig.add_subplot(2,2,4) 
    
    ax_l.plot(time_t, cv_ref[0,:],
              color='r', marker=marker, lw=lwref)   
    
    if cv_md_sim is not None:
        ax_l.plot(time_t, cv_md_sim[0,:],
                  color='b', marker=marker2, lw=lwcos, linestyle='--')   
    
        ax_r = ax_l.twinx()
        
        if compareType=="difference":
            ax_r.semilogy(time_t, np.abs(cv_ref[0,:]-cv_md_sim[0,:])/cv_ref[0,:],
                      'k', lw=lwc)
            ax_r.set_ylabel(r"$\delta(c_{e, -1}^-)/c_{e, -1}^- ._{ref}$")
        else:
            ax_r.plot(time_t, np.abs(cv_md_sim[0,:]/cv_ref[0,:]),
                      'k', lw=lwc)
            ax_r.set_ylabel(r"$c_{e, -1}^-/c_{e, -1}^- ._{ref}$")
        
    ax_l.set_ylabel(r"$c_{e, -1}^- \ (mol/m^3)$")
    ax_l.set_xlabel(r"$t \ (s)$")
    ax_l.grid()
    plt.tight_layout()
    ####################################################################################################################################################
    if savefig:
        plt.savefig(figname, dpi=dpi)
    plt.show()
    


def error_L1(t, y, y_ref, options_electrolyte, options_cathode):
    # In reality dx = dx/L
    dx = np.r_[options_electrolyte['mesh']['cellSize'], options_cathode['activematerial']['mesh']['cellSize'],
               options_cathode['currentcollector']['mesh']['cellSize']]
    dx_const=dx[0] 
    
    # print(y.shape, y_ref.shape) 
    dt = t[1:]-t[:-1]
    t_span = t[-1]-t[0]
    
    err_x = np.zeros(t.size)
    
    for i in range(t.size):
        err_x[i] = np.sum(np.abs((y[:, i]- y_ref[:, i])/(np.abs(y_ref[:, i]) + 1e-8)) * dx_const)    
                  
    err_xt = np.sum(err_x[1:]  * dt)/t_span
    return err_xt    


# Interpret solutions
def get_errors_L1(sol, ref_sol, options_electrolyte, options_cathode):
    err_L1 = []               
    for i in range(len(sol)):
        x_t, time_t, y_aux_a_t, y_c_e_t, y_phi_e_t, cv_ref, y_c_s_t, y_phi_s_t = y_ts.get_y_t(sol[i].t, sol[i].y, options_electrolyte, options_cathode)
       
        ref_sol.y = ref_sol.sol(time_t/options_electrolyte['parameters']['t_c'])
        ref_sol.t = time_t
       
        ref_x_t, ref_time_t, ref_y_aux_a_t, ref_y_c_e_t, ref_y_phi_e_t, ref_cv_ref, ref_y_c_s_t, ref_y_phi_s_t = y_ts.get_y_t(ref_sol.t,
                                                                                                                                 ref_sol.y,
                                                                                                                                 options_electrolyte,
                                                                                                                                 options_cathode)
        
        err = error_L1(time_t, np.r_[y_c_e_t, y_c_s_t],
                          np.r_[ref_y_c_e_t, ref_y_c_s_t],
                          options_electrolyte, options_cathode)

        err_L1.append(err)
        
    return err_L1 
    
def get_cv_t(sol_t, sol_y, options_electrolyte, options_cathode):
    x, time, aux_a, ce, phie, cv, cs, phis = y_ts.get_y_t(sol_t, sol_y, options_electrolyte, options_cathode)
    
    return cv
