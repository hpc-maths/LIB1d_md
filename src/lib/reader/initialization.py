# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:14:01 2022

@author: ali.asad
"""

import numpy as np

from src.lib.solver import my_newton_solver as newton
from src.lib.model import _1d_lib_dfv_model as model
from src.lib.jac_reorder import  jac_reorder as tf
from src.lib.model import aux_system as aux_sys

def getFiniteVolumeMesh(nCells, ndimL_Start, ndimL_End, meshoptions=None):
    if meshoptions is None:
        meshoptions={}
    
    
    xFaces = np.linspace(ndimL_Start, ndimL_End, num=nCells+1) # position of cell-faces
    
    xCells = (xFaces[1:]+xFaces[:-1])/2. # positions of cell-centers 
    
    meshoptions['faceX'] = xFaces # face points
    meshoptions['cellX'] = xCells # cell-center points
    meshoptions['cellSize'] = np.diff(meshoptions['faceX'])
    meshoptions['dxBetweenCellCenters'] = np.diff(meshoptions['cellX'])
    
    assert not any(meshoptions['cellSize']==0.), 'some cells are of size 0...'
    assert not any(meshoptions['cellSize']<0.), 'some cells are of negative size...'
    assert not any(meshoptions['dxBetweenCellCenters']==0.), 'some cells have the same centers...'
    assert np.max(meshoptions['cellSize'])/np.min(meshoptions['cellSize']) < 1e10, 'cell sizes extrema are too different'
        
    return meshoptions
####################################################################################################################################################
def getNonUniformFiniteVolumeMesh(nCells, ndimL_Start, ndimL_End, meshoptions=None, exp_factor=None):
    if meshoptions is None:
        meshoptions={}
        
    mid = (ndimL_Start + ndimL_End) / 2  

    if exp_factor is None:
        exp_factor = 1  # Controls clustering at ends
    
    half_nCells = nCells // 2 + 1    
        
    log_half = np.logspace(-exp_factor, 0, half_nCells, base=10.0)
    
    xFaces_half = (log_half - log_half.min()) / (log_half.max() - log_half.min()) * (mid - ndimL_Start) + ndimL_Start

    xFaces = np.concatenate((xFaces_half, 2 * mid - xFaces_half[::-1][1:])) # position of cell-faces
    
    xCells = (xFaces[1:]+xFaces[:-1])/2. # positions of cell-centers 
    
    meshoptions['faceX'] = xFaces # face points
    meshoptions['cellX'] = xCells # cell-center points
    meshoptions['cellSize'] = np.diff(meshoptions['faceX'])
    meshoptions['dxBetweenCellCenters'] = np.diff(meshoptions['cellX'])
    
    assert not any(meshoptions['cellSize']==0.), 'some cells are of size 0...'
    assert not any(meshoptions['cellSize']<0.), 'some cells are of negative size...'
    assert not any(meshoptions['dxBetweenCellCenters']==0.), 'some cells have the same centers...'
    assert np.max(meshoptions['cellSize'])/np.min(meshoptions['cellSize']) < 1e10, 'cell sizes extrema are too different'
        
    return meshoptions

####################################################################################################################################################

def getU_init(options_electrolyte, options_cathode, invertBV=False):
    
    ne = options_electrolyte['mesh']['cellX'].size
    ns = options_cathode['activematerial']['mesh']['cellX'].size + options_cathode['currentcollector']['mesh']['cellX'].size
    
    # Initializing the variables
    c_e = np.ones(ne)
    c_s = (options_cathode['activematerial']['parameters']['c_s_i']/options_cathode['activematerial']['parameters']['c_s_max'])*np.ones(ns)
    
####################################################################################################################################################

    # Initial auxiliary variables
    c_e_0_plus = c_e[0]*np.ones(1)
    c_e_ne_minus = c_e[-1]*np.ones(1)
    c_s_0_plus = c_s[0]*np.ones(1)
   
####################################################################################################################################################
    if (options_cathode['parameters']['ChargeType'] == "CC"):    
        if invertBV:
            from src.lib.model import ocp_graphite as Uocp
            F = options_electrolyte['parameters']['F']
            R = options_electrolyte['parameters']['R']
            T = options_electrolyte['parameters']['T']
            k_0_C = options_electrolyte['parameters']['k_0_C']
            i_0_Li = options_electrolyte['parameters']['i_0_Li']
            phi_c = options_electrolyte['parameters']['phi_c']
            c_e_c = options_electrolyte['parameters']['c_e_c']
            
            c_s_max = options_cathode['activematerial']['parameters']['c_s_max']
            c_s_c = options_cathode['activematerial']['parameters']['c_s_c']
            i_1C = options_cathode['parameters']['i_1C']
            xi = options_cathode['parameters']['xi']
            
            #electrolyte potential
            phi_e = -(2*R*T/F)*np.arcsinh(-xi*i_1C/(2.0*i_0_Li))*np.ones(ne)/phi_c
            phi_e_0_plus = phi_e[0]*np.ones(1)
            phi_e_ne_minus = phi_e[-1]*np.ones(1)
            
            # phi_e_0_plus = -(2*params.R*params.T/params.F)*np.arcsinh(-params.xi*params.i_1C/(2.0*params.i_0_Li))/params.phi_c
            # phi_e = (params.xi*params.i_1C*params.L_c/params.phi_c)*xe + phi_e_0_plus
            # phi_e_ne_minus = (params.xi*params.i_1C*params.L_c/params.phi_c)*0.5 + phi_e_0_plus
            
            # electrode potential
            i_0_C = F*k_0_C*((c_e_ne_minus*c_e_c)**(0.5))*((c_s_0_plus*c_s_c)**(0.5))*((c_s_max-c_s_0_plus*c_s_c)**(0.5))
            phi_s = (Uocp.Ueq(c_s_0_plus) + phi_e[-1]*phi_c + (2*R*T/F)*np.arcsinh(xi*i_1C/(2.0*i_0_C)))*np.ones(ns)/phi_c
            phi_s_0_plus = phi_s[0]*np.ones(1)
            
            # phi_s_0_plus = (Uocp.Ueq(c_s_0_plus) + phi_e[-1]*params.phi_c + (2*params.R*params.T/params.F)*np.arcsinh(params.xi*params.i_1C/(2.0*i_0_C)))/params.phi_c
            # phi_s = params.xi*params.i_1C*params.L_c/(params.phi_c*params.sigma_am*params.sigma_c)*(xs-0.5) + phi_s_0_plus
            # phi_s[params.nam:] =  (phi_s[params.nam:]-phi_s_0_plus)/params.sigma_cc + phi_s[params.nam-1]       
                    
        else:
            phi_e = np.zeros(ne)
            phi_e_0_plus = phi_e[0]*np.ones(1)
            phi_e_ne_minus = phi_e[-1]*np.ones(1)
            
            phi_s = np.ones(ns)
            phi_s_0_plus = phi_s[0]*np.ones(1)
            
        # Initializing algebraic variables to evaluate better initial guess for the Newton solver 
        # U_i_alg = np.r_[phi_e, phi_s, np.r_[c_e_0_plus, phi_e_0_plus, c_e_ne_minus, phi_e_ne_minus, c_s_0_plus, phi_s_0_plus]]
        # U_i_alg_e = np.r_[np.r_[c_e_0_plus, phi_e_0_plus], phi_e, np.r_[c_e_ne_minus, phi_e_ne_minus]]
        # U_i_alg_s = np.r_[np.r_[c_s_0_plus, phi_s_0_plus], phi_s]
        
        U_i_alg = np.r_[np.r_[c_e_0_plus, phi_e_0_plus], phi_e, np.r_[c_e_ne_minus, phi_e_ne_minus, c_s_0_plus, phi_s_0_plus], phi_s]
        
        #sol_alg = root(model.func_U_alg, U_i_alg)
        print("----------------------------------\nAlgebraic system being solved for initialization")
        
        if invertBV:
            print("-- Potentials are defined by inverting BV current")
        else :
            print("-- Potentials are arbitrary (0 and 1) : Brut start")
        
        sol_alg = newton.root_solve(func=lambda x: model.func_U_alg(x, options_electrolyte, options_cathode),
                                    x0 = U_i_alg,
                                    atol = 1e-10,
                                    rtol = 1e-9,
                                    verbose = True)
        
        # sol_alg_e = newton.root_solve(func=lambda x:model.func_U_alg_e(x, c_s_0_plus, phi_s_0_plus, phy_options, sim_options),
        #                             x0 = U_i_alg_e,
        #                             atol = 1e-6,
        #                             rtol = 1e-5,
        #                             verbose = True)
        
        # sol_alg_s = newton.root_solve(func=lambda x:model.func_U_alg_s(x, c_e_ne_minus, phi_e_ne_minus, phy_options, sim_options),
        #                             x0 = U_i_alg_s,
        #                             atol = 1e-6,
        #                             rtol = 1e-5,
        #                             verbose = True)
        
        print("Solution converged for the alegebraic system \n----------------------------------")
        
        c_e_0_plus = sol_alg[0]
        phi_e_0_plus = sol_alg[1]
        
        phi_e = sol_alg[2:ne+2]
        
        c_e_ne_minus = sol_alg[ne+2]
        phi_e_ne_minus = sol_alg[ne+3]
        c_s_0_plus = sol_alg[ne+4]
        phi_s_0_plus = sol_alg[ne+5]
        
        phi_s = sol_alg[ne+6:]
        
        # phi_e = sol_alg[:ne]
        # phi_s = sol_alg[ne:ne+ns]
        
        # c_e_0_plus = sol_alg[-6]
        # phi_e_0_plus = sol_alg[-5]
        # c_e_ne_minus = sol_alg[-4]
        # phi_e_ne_minus = sol_alg[-3]
        # c_s_0_plus = sol_alg[-2]
        # phi_s_0_plus = sol_alg[-1]
    
####################################################################################################################################################
    else:
        # raise Exception('Cannot initialize at constant voltage condition')
        
        phi_e = 0.1*np.ones(ne)
        phi_e_0_plus = phi_e[0]*np.ones(1)
        phi_e_ne_minus = phi_e[-1]*np.ones(1)
        
        phi_c = options_electrolyte['parameters']['phi_c']
        phis_L = options_cathode['parameters']['phi_s_L'](0.0)/phi_c
        phi_s = phis_L*np.ones(ns)
        phi_s_0_plus = phi_s[0]*np.ones(1)
        
        ise = -1.0
        from src.lib.model import bv_current as bv
        cec = options_electrolyte['parameters']['c_e_c']
        phic = options_electrolyte['parameters']['phi_c']
        csc = options_cathode['activematerial']['parameters']['c_s_c']
        
        # x = [0:c_e_ne_minus, 1:phi_e_ne_minus, 2:c_s_0_plus,  3:phi_s_0_plus, 4:ise]         
        # U_5 = np.r_[c_e_ne_minus, phi_e_ne_minus, c_s_0_plus, phi_s_0_plus, ise]
        # f_5 = lambda x: np.r_[aux_sys.G_aux_c(t=0.0, ce=c_e[-1], phie=phi_e[-1], cs=c_s[0], phis=phi_s[0],
        #                                           v_aux_c=np.r_[x[0], x[1], x[2], x[3]],
        #                                           options_electrolyte=options_electrolyte,
        #                                           options_cathode=options_cathode,
        #                                           alg=True),
        #                       x[4] - bv.ise_C(x[0]*cec,
        #                                       x[1]*phic,
        #                                       x[2]*csc,
        #                                       x[3]*phic,
        #                                       options_electrolyte,
        #                                       options_cathode) ]
        # # sol 5 var strategy
        # sol_5 = newton.root_solve(func = f_5,
        #                             x0 = U_5,
        #                             atol = 1e-6,
        #                             rtol = 1e-5,
        #                             verbose = True)
        
        # c_e_ne_minus=sol_5[0]
        # phi_e_ne_minus=sol_5[1]
        # c_s_0_plus=sol_5[2]
        # phi_s_0_plus=sol_5[3]
        
        # phi_e = phi_e_ne_minus*np.ones(ne)
        # phi_e_0_plus = phi_e_ne_minus    
        # c_e = c_e_ne_minus*np.ones(ne)
        # c_e_0_plus = c_e_ne_minus    
        # c_s = c_s_0_plus*np.ones(ns)
        
        # x = [0:c_e_0_plus, 1:phi_e_0_plus, 2:c_e_ne_minus, 3:phi_e_ne_minus, 4:c_s_0_plus,  5:phi_s_0_plus, 6:ise]         
        U_7 = np.r_[c_e_0_plus, phi_e_0_plus, c_e_ne_minus, phi_e_ne_minus, c_s_0_plus, phi_s_0_plus]#, ise]
        f_7 = lambda x: np.r_[aux_sys.G_aux_a(c1=c_e[0], phi1=phi_e[0],
                                                  v_aux_a=np.r_[x[0] ,x[1]],
                                                  options_electrolyte=options_electrolyte),
                              aux_sys.G_aux_c(t=0.0, ce=c_e[-1], phie=phi_e[-1], cs=c_s[0], phis=phi_s[0],
                                                  v_aux_c=np.r_[x[2], x[3], x[4], x[5]],
                                                  options_electrolyte=options_electrolyte,
                                                  options_cathode=options_cathode,
                                                  alg=True)]
                              # x[6] - bv.ise_A(x[0]*cec,
                              #                 x[1]*phic,
                              #                 options_electrolyte) ]

                              # x[6] - bv.ise_C(x[2]*cec,
                              #                 x[3]*phic,
                              #                 x[4]*csc,
                              #                 x[5]*phic,
                              #                 options_electrolyte,
                              #                 options_cathode) ]
        
        # sol 7 var strategy
        sol_7 = newton.root_solve(func = f_7,
                                    x0 = U_7,
                                    atol = 1e-6,
                                    rtol = 1e-5,
                                    verbose = True)
        
        print("[0:c_e_0_plus, 1:phi_e_0_plus, 2:c_e_ne_minus, 3:phi_e_ne_minus, 4:c_s_0_plus,  5:phi_s_0_plus, 6:ise]")
        print(sol_7)
        
        c_e_0_plus=sol_7[0]
        phi_e_0_plus=sol_7[1]
        c_e_ne_minus=sol_7[2]
        phi_e_ne_minus=sol_7[3]
        c_s_0_plus=sol_7[4]
        phi_s_0_plus=sol_7[5]
        
        phi_e = 0.5*(phi_e_0_plus + phi_e_ne_minus)*np.ones(ne)
        # c_e = 0.5*(c_e_0_plus + c_e_ne_minus)*np.ones(ne)
        # c_s = c_s_0_plus*np.ones(ns)
        phi_s = phi_s_0_plus*np.ones(ns)
                
                        
# Definition of U
    U_aux_e_a_i = tf.gather_Xi(c_e_0_plus, phi_e_0_plus)
    U_e_i = tf.gather_Xi(c_e, phi_e)
    U_aux_e_c_i = np.r_[c_e_ne_minus, phi_e_ne_minus]
    U_aux_s_c_i = np.r_[c_s_0_plus, phi_s_0_plus]
    U_s_i = tf.gather_Xi(c_s, phi_s)

    U_e_i = np.r_[U_aux_e_a_i, U_e_i, U_aux_e_c_i]
    U_s_i = np.r_[U_aux_s_c_i, U_s_i]

    mass_diag_e_i = np.r_[np.zeros(2), tf.gather_Xi(np.ones(ne), np.zeros(ne)), np.zeros(2)] 
    mass_diag_s_i = np.r_[np.zeros(2), tf.gather_Xi(np.ones(ns), np.zeros(ns))]
    
    return U_e_i, mass_diag_e_i, U_s_i, mass_diag_s_i
####################################################################################################################################################
    