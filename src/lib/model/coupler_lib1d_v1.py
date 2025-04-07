# -*- coding: utf-8 -*-
"""
Created on Mon Aug  22 16:19:59 2022

@author: laurent.francois and ali.asad
"""

import numpy as np

from src.lib.rhapsopy import coupling_v1 as coupling

from src.lib.model import _1d_lib_dfv_model as model
from src.lib.model import bv_current as bv

import scipy.sparse as sparse

import logging


INTEGRATION_DETAIL = 20
INTEGRATION_DETAIL2 = 15
INTEGRATION = 30



# np.seterr(divide="raise")


# Problem wrapper / Coupler
class Coupler(coupling.BaseCoupler):
    """ This class is the interface between the generic co-simulation class and the subsystem solvers.
    It is physics-aware. """
    def __init__(self, options1, options2, coupling_modes):

        # Create logger object
        self.logger = logging.getLogger('LIB.coupler')
        self.logger.handlers = [] # drop previous handlers
        self.logger.setLevel(100) # no logging by default
        
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.log(INTEGRATION_DETAIL, 'Initialising coupler object')


        # options structures for the subsytems
        self.options = [options1, options2] # option structure for both subsystems
        self.options1 = options1
        self.options2 = options2
        self.nSubsystems = 2

        if self.options1['sim_type']=="md_sim_flux" and self.options2['sim_type']=="md_sim_flux":
           self.nCouplingVars = 2 # number of coupling variables
        elif self.options1['sim_type']=="md_coupling_vars" and self.options2['sim_type']=="md_coupling_vars":
           self.nCouplingVars = 4 # number of coupling variables
        else:
            raise Exception("No coupling condition specified")

        
        self.nx = [opts['nCells'] for opts in self.options]
        self.coupling_modes = coupling_modes
        self.predictors = None # Dictionnary form of the polynomial predictors used by the co-simulation class, for better handling

        # waveform relaxation on predictors or on dense outputs directly (more precise)
        self.bUseDenseOutput=False
        
        # Choice of integration for each subproblem
        self.adaptive_subsolves = False #True # if True, adaptative time stepping is used for each solver
        self.rtol_subsolves_default = 1e-9
        self.rtol_getCouplingVars_default = 1e-9
        
        # Analytics
        self.nfev = np.zeros(self.nSubsystems)
    
    def _spreadPredictors(self, pred_list):
        """ Take the generic list of predictors from the co-simulation class, and organise it into a physics-aware dictionary """
        self.logger.log(INTEGRATION_DETAIL, 'spreading predictors')
        
        if self.options1['sim_type']=="md_sim_flux" and self.options2['sim_type']=="md_sim_flux":
           self.predictors = {'coupling_flux': pred_list[0],
                               'coupling_var_phi_s' : pred_list[1]}
           
        elif self.options1['sim_type']=="md_coupling_vars" and self.options2['sim_type']=="md_coupling_vars":
            self.predictors = {'ce':    pred_list[0], # c_e_ne_minus
                               'phie':  pred_list[1], # phi_e_ne_minus
                               'cs':  pred_list[2], # c_s_0_plus
                               'phis': pred_list[3]} # phi_s_0_plus
        else:
            raise Exception("No coupling condition specified")            
    
    def _feedOptions(self): 
        
        self.logger.log(INTEGRATION_DETAIL, 'feeding options')
        
        if self.options1['sim_type']=="md_sim_flux" and self.options2['sim_type']=="md_sim_flux":
           self.options1['coupling_flux'] = lambda t : self.predictors['coupling_flux'].eval_single(t)
           self.options2['coupling_flux'] = lambda t : self.predictors['coupling_flux'].eval_single(t)
           self.options2['coupling_var_phi_s'] = lambda t : self.predictors['coupling_var_phi_s'].eval_single(t)
        elif self.options1['sim_type']=="md_coupling_vars" and self.options2['sim_type']=="md_coupling_vars":
           self.options1['coupling_vars'] = lambda t : np.r_[self.predictors['ce'].eval_single(t), self.predictors['phie'].eval_single(t)]
           self.options2['coupling_vars'] = lambda t : np.r_[self.predictors['cs'].eval_single(t), self.predictors['phis'].eval_single(t)]
        else:
            raise Exception("No coupling condition specified")
        
    def getCouplingVars(self,t,y):
        """ Call each system's object to gather the required coupling variables """
        self.logger.log(INTEGRATION_DETAIL, 'getting coupling variables')
        
        ne = self.options1['nCells']
        coupling_var = y[2*ne+2:2*ne+6]
                
        if self.options1['sim_type']=="md_sim_flux" and self.options2['sim_type']=="md_sim_flux":
                       
            from src.lib.model import aux_system as flx_aux
            ce = y[2*ne+2]
            phie = y[2*ne+3]
            cs = y[2*ne+4]
            phis = y[2*ne+5]
            fun = lambda x: flx_aux.G_aux_c(t=t, ce=ce, phie=phie,
                                            cs=cs, phis=phis, v_aux_c=x,
                                            options_electrolyte=self.options1,
                                            options_cathode=self.options2, alg=True)
           
            from src.lib.solver import my_newton_solver as root
            rtol = self.rtol_getCouplingVars_default
            coupling_var = root.root_solve(func=fun, x0=coupling_var,
                                            atol=rtol/10., rtol=rtol, verbose=False)   # tolerances for subsystem should be objects of coupler
            
           
            phis0 = coupling_var[3]
                       
            cem = coupling_var[0]*self.options1['parameters']['c_e_c']
            phiem = coupling_var[1]*self.options1['parameters']['phi_c']
            csp = coupling_var[2]*self.options2['activematerial']['parameters']['c_s_c']
            phisp = coupling_var[3]*self.options1['parameters']['phi_c']
            ise_C_flux = bv.ise_C(cem, phiem, csp, phisp, self.options1, self.options2)
          
            return np.r_[ise_C_flux, phis0] 
        
        elif self.options1['sim_type']=="md_coupling_vars" and self.options2['sim_type']=="md_coupling_vars":
            from src.lib.model import aux_system as flx_aux
            ce = y[2*ne]
            phie = y[2*ne+1]
            cs = y[2*ne+6]
            phis = y[2*ne+7]
            fun = lambda x: flx_aux.G_aux_c(t=t, ce=ce, phie=phie,
                                            cs=cs, phis=phis, v_aux_c=x,
                                            options_electrolyte=self.options1,
                                            options_cathode=self.options2, alg=True)
            
            from src.lib.solver import my_newton_solver as root
            rtol = self.rtol_getCouplingVars_default
            coupling_var = root.root_solve(func=fun, x0=coupling_var,
                                            atol=rtol/10, rtol=rtol,
                                            verbose=False)   # tolerances for subsystem should be objects of coupler
            
            return coupling_var
        else:
            raise Exception("No coupling condition specified")
                 
    def integrateSingleSubsystem(self, isolv, t0, y0, dt, preds, last_outs, rtol=None, bDebug=False):
        """ Performs one iteration of a co-simulation step (Jacobi or Gauss-Seidel)
              --> computes the value of the overall state vector at time t+dt,
                  starting from state y at time t. """
        self.logger.log(INTEGRATION_DETAIL, f'coupler: integrating subsystem {isolv}')
        
        self._spreadPredictors(preds) #1
        
        self._feedOptions() #2

        #TODO 1 and 2 may be combined in 1 later.

        if rtol is None:
            rtol = self.rtol_subsolves_default
        atol = rtol/10

        if self.adaptive_subsolves:
                
            from scipy.integrate import solve_ivp
            from src.lib.solver.radau_dae import RadauDAE
    
            method=RadauDAE
            bPrint=False
            bPrintProgress=False      
        
            integrator = lambda fun, y0, mass, sparsity, var_idx: solve_ivp(fun=fun, y0=y0,
                                t_span=[t0,t0+dt], max_step=dt, first_step=dt/2,
                                rtol=rtol, atol=atol, 
                                max_newton_ite=10, max_bad_ite=2,
                                var_index = var_idx,
                                jac=None, jac_sparsity=sparsity,
                                method=method, vectorized=False,
                                dense_output=self.bUseDenseOutput,
                                mass=mass, bPrint=bPrint,
                                scale_residuals = True,
                                scale_newton_norm = True,
                                scale_error = True,
                                bPrintProgress=bPrintProgress)
        else:
            from src.lib.solver import my_newton_solver as solve_dae
            integrator = lambda fun, y0, mass, sparsity: solve_dae.root_solve(func=fun,
                                                           x0=y0, sparsity=sparsity,
                                                           atol=atol, rtol=rtol, verbose=False) 

        ## get each subsystem's state vector
        ne = self.options1['nCells']
        ye = y0[:2*ne+4]
        ys = y0[2*ne+4:]
                        
        ns = self.options2['nCells']
        
        # Sparsity pattern 
        uband=6; lband=-uband
        offsets = [i for i in range(lband,uband)]
         
        if (isolv == 0):
            sparsity_pattern = sparse.diags(diagonals=[np.ones((2*ne+4 - abs(i))) for i in offsets], offsets=offsets, format='csc')
            mass_np = np.diag(self.options1['mass'])
            mass = sparse.csc_matrix(mass_np)
            var_idx = abs( (self.options1['mass']==0)*1 - 1)
            y0  = ye            
            
            if self.adaptive_subsolves:
                fun = lambda t,x: model.func_U_e(t=t, u=x,
                                                 options_electrolyte=self.options1,
                                                 options_cathode=self.options2)
            else:
                fun = lambda x: np.dot(mass, (x - y0)) - dt*np.array(model.func_U_e(t=t0+dt,
                                                                                    u=x,
                                                                                    options_electrolyte=self.options1,
                                                                                    options_cathode=self.options2))
        else:
            sparsity_pattern = sparse.diags(diagonals=[np.ones((2*ns+2 - abs(i))) for i in offsets], offsets=offsets, format='csc')
            mass_np = np.diag(self.options2['mass'])
            mass = sparse.csc_matrix(mass_np)
            var_idx = abs( (self.options2['mass']==0)*1 - 1)
            y0  = ys
            
            if self.adaptive_subsolves:
                fun = lambda t,x: model.func_U_s(t=t, u=x,
                                                 options_electrolyte=self.options1,
                                                 options_cathode=self.options2)
            else:
                fun = lambda x: np.dot(mass, (x - y0)) - dt*np.array(model.func_U_s(t=t0+dt,
                                                                                    u=x,
                                                                                    options_electrolyte=self.options1,
                                                                                    options_cathode=self.options2))

        current_out = integrator(fun=fun, y0=y0, mass=mass,
                                 sparsity=sparsity_pattern, var_idx=var_idx)
        
        
        # print(current_out.success, current_out.message)
        if self.adaptive_subsolves:    
            self.nfev[isolv] += current_out.nlu
            
            if not current_out.success:
                # import pdb; pdb.set_trace()
                from src.lib.rhapsopy.rhapsopy_utils import ExceptionWhichMayDisappearWhenLoweringDeltaT, WRnonConvergence
                raise ExceptionWhichMayDisappearWhenLoweringDeltaT(current_out.message)
                # raise Exception(current_out.message)
        else:
            class set_current_out:
                def __init__(self, y, t):
                    self.y = y
                    self.t = t
            current_out = set_current_out(np.array(current_out).reshape(current_out.size, 1), t0+dt)
            
        # construct overall time derivatives
        self.logger.log(INTEGRATION_DETAIL, f'integration completed {isolv}')
        
        return current_out