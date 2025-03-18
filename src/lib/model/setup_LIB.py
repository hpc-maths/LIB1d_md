# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:29:47 2022

@author: ali.asad
"""

#add battery type in simulation json for half cell or full cell simulations

#import numpy as np

from src.lib.reader import  initialization as setuptool

def getSetup(phy_options, sim_options, invertBV=False, half_cell=True, aux_sys=True, bUnfiromMesh=True, exp_factor=None):
    
    L_c = phy_options['Electrolyte']['L_e']['value']+phy_options['Electrode']['L_am']['value']+phy_options['CurrentCollector']['L_cc']['value']
    ##################################################################################################################    
    
    options_electrolyte = {}
    options_electrolyte['mesh'] = {}
    options_electrolyte['parameters'] = {}
    
    nCells_e = sim_options['Mesh']['ne']['value']
    L_Start_e = 0.0
    L_End_e = phy_options['Electrolyte']['L_e']['value']/L_c    
    
    if bUnfiromMesh:
        options_electrolyte['mesh'] = setuptool.getFiniteVolumeMesh(nCells_e, L_Start_e, L_End_e)
    else:
        options_electrolyte['mesh'] = setuptool.getNonUniformFiniteVolumeMesh(nCells_e, L_Start_e, L_End_e, exp_factor=exp_factor)
    
    options_electrolyte['parameters']['c_e_i'] = phy_options['Electrolyte']['c_e_i']['value']
    options_electrolyte['parameters']['c_e_c'] = options_electrolyte['parameters']['c_e_i']
    options_electrolyte['parameters']['D_e_c'] = phy_options['Electrolyte']['D_e_c']['value']
    options_electrolyte['parameters']['kappa_e_c'] = phy_options['Electrolyte']['kappa_e_c']['value']
    options_electrolyte['parameters']['t_0_plus'] = phy_options['Electrolyte']['t_0_plus']['value']
    options_electrolyte['parameters']['del_e'] = phy_options['Electrolyte']['del_e']['value']
    options_electrolyte['parameters']['i_0_Li'] = phy_options['ButlerVolmerParams']['i_0_Li']['value']
    options_electrolyte['parameters']['k_0_C'] = phy_options['ButlerVolmerParams']['k_0_C']['value']/phy_options['Constants']['F']['value']
    options_electrolyte['parameters']['F'] = phy_options['Constants']['F']['value']
    options_electrolyte['parameters']['R'] = phy_options['Constants']['R']['value']
    options_electrolyte['parameters']['T'] = phy_options['Constants']['T']['value']
    
    # Derived scales
    phi_c = options_electrolyte['parameters']['R']*options_electrolyte['parameters']['T']/options_electrolyte['parameters']['F']
    t_c = L_c**2/options_electrolyte['parameters']['D_e_c']
    
    options_electrolyte['parameters']['phi_c'] = phi_c
    options_electrolyte['parameters']['t_c'] = t_c
    options_electrolyte['parameters']['L_c'] = L_c
    options_electrolyte['parameters']['inv_i_e_c'] = L_c/(phi_c*options_electrolyte['parameters']['kappa_e_c'])   
    options_electrolyte['parameters']['inv_N_e_c'] = L_c/(options_electrolyte['parameters']['c_e_c']*options_electrolyte['parameters']['D_e_c'])
    
    options_electrolyte['parameters']['Lambda'] = 2.*phi_c*(1.-options_electrolyte['parameters']['t_0_plus'])*(1.+options_electrolyte['parameters']['del_e']) 
    
    beta_n = options_electrolyte['parameters']['Lambda']*options_electrolyte['parameters']['kappa_e_c']*(1.-options_electrolyte['parameters']['t_0_plus'])
    beta_d = options_electrolyte['parameters']['D_e_c']*options_electrolyte['parameters']['c_e_c']*options_electrolyte['parameters']['F'] 
    options_electrolyte['parameters']['beta'] = beta_n/beta_d
    
    # Simulation parameters
    options_electrolyte['parameters']['D_e'] = 1.
    options_electrolyte['parameters']['kappa_e'] = 1. 
    ##################################################################################################################

    options_cathode = {}
    options_cathode['activematerial'] = {'mesh':{}, 'parameters':{}}
    options_cathode['currentcollector'] = {'mesh':{}, 'parameters':{}}
    options_cathode['parameters'] = {}
    
    nCells_am = sim_options['Mesh']['nam']['value']
    L_Start_am = phy_options['Electrolyte']['L_e']['value']/L_c
    L_End_am = phy_options['Electrolyte']['L_e']['value']/L_c+phy_options['Electrode']['L_am']['value']/L_c # Change to Cathode
    
    if bUnfiromMesh:
        options_cathode['activematerial']['mesh'] = setuptool.getFiniteVolumeMesh(nCells_am, L_Start_am, L_End_am)
    else:
        options_cathode['activematerial']['mesh'] = setuptool.getNonUniformFiniteVolumeMesh(nCells_am, L_Start_am, L_End_am, exp_factor=exp_factor)
    
    options_cathode['activematerial']['nCells'] = nCells_am
    
    options_cathode['activematerial']['parameters']['c_s_max'] = phy_options['Electrode']['c_s_max']['value']
    options_cathode['activematerial']['parameters']['c_s_i'] = sim_options['ChargeRate']['c_s_i']['value']
    options_cathode['activematerial']['parameters']['c_s_c'] = options_cathode['activematerial']['parameters']['c_s_max']
    options_cathode['activematerial']['parameters']['D_am_c'] = phy_options['Electrode']['D_am_c']['value']
    options_cathode['activematerial']['parameters']['sigma_am_c'] = phy_options['Electrode']['sigma_am_c']['value']
    
    nCells_cc = sim_options['Mesh']['ncc']['value']
    L_Start_cc = phy_options['Electrolyte']['L_e']['value']/L_c+phy_options['Electrode']['L_am']['value']/L_c
    L_End_cc = 1.0 # Change to Cathode
    
    if bUnfiromMesh:
        options_cathode['currentcollector']['mesh'] = setuptool.getFiniteVolumeMesh(nCells_cc, L_Start_cc, L_End_cc)
    else:
        options_cathode['currentcollector']['mesh'] = setuptool.getNonUniformFiniteVolumeMesh(nCells_cc, L_Start_cc, L_End_cc, exp_factor=exp_factor)
    
    options_cathode['currentcollector']['nCells'] = nCells_cc
    
    options_cathode['currentcollector']['parameters']['sigma_cc_c'] = phy_options['CurrentCollector']['sigma_cc_c']['value']
    
    # Derived scales for electrode
    options_cathode['parameters']['sigma_c'] = options_cathode['activematerial']['parameters']['sigma_am_c']
    options_cathode['parameters']['phi_c'] = phi_c
    options_cathode['parameters']['inv_i_s_c'] = L_c/(phi_c*options_cathode['parameters']['sigma_c'])   
    options_cathode['parameters']['inv_N_s_c'] = L_c/(options_cathode['activematerial']['parameters']['c_s_c']*options_cathode['activematerial']['parameters']['D_am_c'])
    options_cathode['parameters']['varepsln'] = options_cathode['activematerial']['parameters']['D_am_c']/options_electrolyte['parameters']['D_e_c']
    options_cathode['parameters']['F'] = phy_options['Constants']['F']['value']
    # Simulation parameters
    options_cathode['activematerial']['parameters']['D_am'] = 1.
    options_cathode['activematerial']['parameters']['sigma_am'] = 1.
    options_cathode['currentcollector']['parameters']['sigma_cc'] = options_cathode['currentcollector']['parameters']['sigma_cc_c']/options_cathode['parameters']['sigma_c'] 
    # Operating condition
    options_cathode['parameters']['ChargeType'] = sim_options['ChargeRate']['type']['value']
    options_cathode['parameters']['i_1C'] = options_electrolyte['parameters']['F']*options_cathode['activematerial']['parameters']['c_s_max']*phy_options['Electrode']['L_am']['value']/3600.0
    options_cathode['parameters']['xi'] = sim_options['ChargeRate']['xi']['value']
    options_cathode['parameters']['phi_s_L'] = sim_options['ChargeRate']['phi_s_L']['value']
    #####################################################################################################################
    # Initial solution at initial time 
    options_electrolyte['nCells'] = nCells_e
    options_cathode['nCells'] = nCells_am + nCells_cc
        
    Uei, Mei, Usi, Msi = setuptool.getU_init(options_electrolyte, options_cathode, invertBV=invertBV)
    
    options_electrolyte['y0'] = Uei
    options_electrolyte['mass'] = Mei
    
    options_cathode['y0'] = Usi
    options_cathode['mass'] = Msi
        
    if not half_cell:
        options_anode ={}
        ## to be developed later

    if half_cell:
        return options_electrolyte, options_cathode
    else:
        return options_anode, options_electrolyte, options_cathode