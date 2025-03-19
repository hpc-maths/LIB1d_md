# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:15:20 2023

@author: ali.asad
"""
# %%
import numpy as np

# customs modules
from src.lib.reader import  reader as params
from src.lib.model import setup_LIB as setup 

from src.lib.model import _1d_lib_dfv_model as coupled_model

# from src.lib.model.coupler_lib1d_rhapsopy import BaseCoupler
# from src.lib.rhapsopy.coupling import Orchestrator
# from src.lib.rhapsopy.accelerators import NewtonSolver, DampedNewtonSolver, IQNSolver, AitkenUnderrelaxationSolver, AitkenScalarSolver, FixedPointSolver, AndersonSolver, ExplicitSolver


from scipy.integrate import solve_ivp
from src.lib.solver.radau_dae import RadauDAE
import scipy.sparse as sparse

from tqdm import tqdm

# Function for the initial CC mode simulation. 
def initial_CC_sim(tini,
                   t_CC_end,
                   y0_cc,
                   rtol=1e-10,
                   options_electrolyte=None,
                   options_cathode=None
                   ):
    
    if (options_electrolyte is None):
        raise Exception('options_electrolyte not provided')
    if (options_cathode is None):
        raise Exception('options_cathode not provided')
    
    method=RadauDAE
    atol = rtol/10.
    bPrint=False

    options_cathode['sim_type']="monolithic"
    options_electrolyte['sim_type']="monolithic"
    Mi = sparse.csc_matrix(np.diag(np.r_[options_electrolyte['mass'], options_cathode['mass']]))
    var_idx = abs( (np.r_[options_electrolyte['mass'], options_cathode['mass']]==0 )*1 - 1)

    # sparsity pattern
    uband=6; lband=-uband
    offsets = [i for i in range(lband,uband)]
    sparsity_pattern = sparse.diags(diagonals=[np.ones((y0_cc.size - abs(i))) for i in offsets], offsets=offsets, format='csc')

    options_cathode['parameters']['ChargeType'] = "CC"
    options_cathode['parameters']['xi']  = 1.

    out_cc = solve_ivp(fun = lambda t, u:coupled_model.func_U(u, t, options_electrolyte, options_cathode),
                        t_span=(tini, t_CC_end),
                        y0=y0_cc,
                        max_step=np.inf,
                        rtol=rtol, atol=atol,
                        max_newton_ite=10, max_bad_ite=2,
                        var_index = var_idx,
                        jac=None, jac_sparsity=sparsity_pattern,
                        method=method, vectorized=False, first_step=1e-8, dense_output=True,
                        mass=Mi, bPrint=bPrint)

    print('Initial CC solution computed')      
    print('  --> "{}"'.format(out_cc.message))
    if not out_cc.success:
        raise Exception('Initial CC solution failed')
    
    return out_cc

# Function to perform multi-domain simulations
from scipy.optimize import OptimizeResult as OdeResult
def perform_md_simulation(y0_global, t_vec,
                          order=None,
                          bExplicitCoupling=True,
                          NITER_MAX=100,
                          options_electrolyte=None,
                          options_cathode=None,
                          adaptive_subsolves=True,
                          adaptive_md_sim=False,
                          dt_rtol=None,
                          bmd_simVersion=False,
                          md_sim_logger=100,
                          coupler_logger=100,
                          outRef=None):
    
    if (options_electrolyte is None):
        raise Exception('options_electrolyte not provided')
    if (options_cathode is None):
        raise Exception('options_cathode not provided')
        
    toldict = {1: 1e-7, 2:1e-9, 3:1e-9, 4: 1e-10, 5:1e-10} # to help with convergence of lower-order schemes
    tol = toldict[order]
    subsolve_tol = tol/5. # rtol for integration of subsystems
    getCV_tol = subsolve_tol/5. # rtol for synchronization step
    WR_tol = tol 
    
    
    if not bmd_simVersion:
        from src.lib.model.coupler_lib1d_rhapsopy import BaseCoupler
        from src.lib.rhapsopy.coupling import Orchestrator
        from src.lib.rhapsopy.accelerators import NewtonSolver, DampedNewtonSolver, IQNSolver, AitkenUnderrelaxationSolver, AitkenScalarSolver, FixedPointSolver, AndersonSolver, ExplicitSolver
    
        coupler = BaseCoupler(options_electrolyte, options_cathode, coupling_modes=['neumann', 'neumann'])        
        coupler.adaptive_subsolves = adaptive_subsolves
        coupler.rtol_getCouplingVars_default = getCV_tol
        coupler.rtol_subsolves_default = subsolve_tol
        coupler.logger.setLevel(coupler_logger)
        
        md_sim = Orchestrator(coupler=coupler, order=order)   
        md_sim.subsystem_ordering = [0, 1]
        md_sim.logger.setLevel(md_sim_logger)
        md_sim.gauss_seidel=False
            
        if bExplicitCoupling: # explicit coupling
            md_sim.interfaceSolver = ExplicitSolver
            md_sim.NITER_MAX = NITER_MAX
            md_sim.waveform_tolerance = WR_tol
            md_sim.raise_error_on_non_convergence = False
        else: # implicit coupling
            md_sim.interfaceSolver = FixedPointSolver
            # md_sim.interfaceSolver = DampedNewtonSolver; print('using damped Newton solver !')
            md_sim.NITER_MAX = NITER_MAX
            md_sim.waveform_tolerance = WR_tol
            md_sim.raise_error_on_non_convergence = False 
    else:
        from src.lib.model.coupler_lib1d_v1 import Coupler
        from src.lib.rhapsopy.coupling_v1 import Orchestrator

        coupler = Coupler(options_electrolyte, options_cathode, coupling_modes=['neumann', 'neumann'])
        coupler.adaptive_subsolves = adaptive_subsolves
        coupler.logger.setLevel(coupler_logger)
        coupler.rtol_getCouplingVars_default = getCV_tol
        coupler.rtol_subsolves_default = subsolve_tol
        
        md_sim = Orchestrator(coupler=coupler, NMAX=order)
        md_sim.md_sim_ordering = [0, 1]
        md_sim.logger.setLevel(md_sim_logger)
        
        if bExplicitCoupling: # explicit coupling
            md_sim.interfaceSolver = 'explicit'
            md_sim.NITER_MAX = 1
            md_sim.waveform_tolerance = WR_tol
            md_sim.raise_error_on_non_convergence = False
        else: # implicit coupling
            md_sim.interfaceSolver = 'fixed-point'
            md_sim.NITER_MAX = NITER_MAX
            md_sim.waveform_tolerance = WR_tol
            md_sim.raise_error_on_non_convergence = False
    
    if not adaptive_md_sim:
        md_sim_type = 'basic'
    else:
        md_sim_type = 'adaptive'
    
    nt_HOI = 4 # the first 4 steps are being replaced by the initialisaiton procedure
    
    if (not (outRef is None)): # and (order>1): # initialise using the provided high-accuracy reference solution
    #   print('initialisation from reference solution')
      
      for p in md_sim.preds:
        p.reset()
      reset_predictors = False
      for t in t_vec[:nt_HOI]:
        # md_sim._advancePredictors(t=t, coupling_vars=outRef.solz(t)) # interpolate reference solution
        ne = options_electrolyte['nCells']
        md_sim._advancePredictors(t=t, coupling_vars=outRef.sol(t)[2*ne+2:2*ne+6]) # interpolate reference solution
        
      used_t_vec = t_vec.copy()
      used_t_vec = used_t_vec[nt_HOI:]
      from scipy.interpolate import interp1d
      y0 = interp1d(x=outRef.t, y=outRef.y, kind='cubic')(used_t_vec[0])
      bHOI = False
    else:
    #   print('no initialisation')
      y0 = y0_global
      reset_predictors = True
      used_t_vec = t_vec.copy()
      bHOI = False
      
      
    try :
        if md_sim_type == 'basic':
            # import pdb; pdb.set_trace()
            if not bmd_simVersion:
                out = md_sim.basic_integration(y0=y0, t_vec=used_t_vec,
                                            reset_predictors = reset_predictors, 
                                            high_order_iter_init= bHOI, nt_HOI=nt_HOI,
                                            nDebugAfterNsteps=np.inf)
            else:
                out = md_sim.basic_integration(y0=y0, t_vec=used_t_vec,
                                            reset_predictors = reset_predictors, 
                                            nDebugAfterNsteps=np.inf)
            
        elif md_sim_type == 'adaptive': 
            if dt_rtol is None:
                rtol=1e-7
            else:
                rtol = dt_rtol
            atol=rtol/10
            md_sim.embedded_method=True
            out = md_sim.adaptive_integration(y0=y0, t_span=[used_t_vec[0], used_t_vec[-1]],
                                              rtol=rtol, atol=atol,
                                              reset_predictors=reset_predictors,
                                              first_step=1e-2,
                                              max_step=1.,
                                              bDenseOutput=True)  
        else:
            raise Exception(f'MD simulationtype {md_sim_type} unknown')
            
    except Exception as e:
        raise e
        print('Random exception caught \n', e)   
        out = OdeResult()
        out.success = False
        out.message = f"Rejected Multi-Domain Simulation with Early exit due to {e}"
        out.y = np.nan*np.ones(shape=(np.array(y0_global).T.shape)) ## ATTENTION we use this hack only for conv. study
            
    return out

# Function defining the convergence study loop
def convergence_study_loop(nt_vec,
                           order_vec,
                           tmd_start,
                           tmd_end,
                           y0_global,
                           bExplicitCoupling=True,
                           options_electrolyte=None,
                           options_cathode=None,
                           outRef=None,
                           nparallel=0):
    
    if (options_electrolyte is None):
        raise Exception('options_electrolyte not provided')
    if (options_cathode is None):
        raise Exception('options_cathode not provided')
    
    if nparallel==0: # old sequential
      sim_md_sols = [[None for i in nt_vec] for k in order_vec]
      for i, current_nt in enumerate(tqdm(nt_vec)) :
          t_vec = np.linspace(tmd_start, tmd_end, current_nt)
          for j, current_order in enumerate(order_vec):
              out_md_sim = perform_md_simulation(y0_global=y0_global, t_vec=t_vec,
                                                 order=current_order,
                                                 bExplicitCoupling=bExplicitCoupling,
                                                 options_electrolyte=options_electrolyte,
                                                 options_cathode=options_cathode,
                                                 NITER_MAX=100,
                                                 outRef=outRef,
                                                 bmd_simVersion=True, # coupling code version for paper
                                                 md_sim_logger=100)
      
              # sim_md_sols[j].append(out_md_sim)
              sim_md_sols[j][i] = out_md_sim
    else: # parallel
      
      sim_md_sols = [[None for i in nt_vec] for k in order_vec]
      
      from joblib import Parallel, delayed, parallel_config
      from itertools import product
      
      def parfun(current_nt, current_order):
          t_vec = np.linspace(tmd_start, tmd_end, current_nt)
          return perform_md_simulation(y0_global=y0_global, t_vec=t_vec,
                                       order=current_order,
                                       bExplicitCoupling=bExplicitCoupling,
                                       options_electrolyte=options_electrolyte,
                                       options_cathode=options_cathode,
                                       NITER_MAX=100,
                                       outRef=outRef,
                                       bmd_simVersion=True, # coupling code version for paper
                                       md_sim_logger=100)
              
      data = list(product(nt_vec, order_vec))
      with parallel_config(backend="loky", inner_max_num_threads=2):
          pool = Parallel(n_jobs=nparallel, verbose=1000)
          results = pool(delayed(parfun)(current_nt, current_order) for current_nt, current_order in data)

      # inefficient way to reorganise but works
      for it, ((current_nt, current_order), out) in enumerate(zip(list(data),results)):
        i = nt_vec.tolist().index(current_nt)
        j = order_vec.tolist().index(current_order)
        # print(it,i,j)
        sim_md_sols[j][i] = out
      
      
    print('------------------------------\nstudy loop ended')
    if True:
        for i, current_order in enumerate(order_vec):
            print(f'-------------p={current_order-1}-----------------')
            for j, current_nt in enumerate(nt_vec):
                print(f"nt={current_nt} : {sim_md_sols[i][j].message}")
    
    return sim_md_sols

# Function to obtain the reference solution from monolithic simulation  
def get_ref_sol(y0,
                tstart,
                tend,
                rtol=1e-10,
                options_electrolyte=None,
                options_cathode=None
                ):
    
    if (options_electrolyte is None):
        raise Exception('options_electrolyte not provided')
    if (options_cathode is None):
        raise Exception('options_cathode not provided')
    
    method=RadauDAE
    atol = rtol/10.
    bPrint=False

    assert options_cathode['sim_type']=="monolithic", 'ref sol should be monolithic'
    assert options_electrolyte['sim_type']=="monolithic", 'ref sol should be monolithic'
    
    Mi = sparse.csc_matrix(np.diag(np.r_[options_electrolyte['mass'], options_cathode['mass']]))
    var_idx = abs( (np.r_[options_electrolyte['mass'], options_cathode['mass']]==0 )*1 - 1)

    # sparsity pattern
    uband=6; lband=-uband
    offsets = [i for i in range(lband,uband)]
    sparsity_pattern = sparse.diags(diagonals=[np.ones((y0.size - abs(i))) for i in offsets], offsets=offsets, format='csc')
        
    assert options_cathode['parameters']['ChargeType'] == "CV", 'Ref sol only for CV'

    ref_sol = solve_ivp(fun = lambda t, u:coupled_model.func_U(u, t, options_electrolyte, options_cathode),
                        t_span=(tstart, tend),
                        y0=y0,
                        max_step=np.inf,
                        rtol=rtol, atol=atol,
                        max_newton_ite=10, max_bad_ite=2,
                        var_index = var_idx,
                        jac=None, jac_sparsity=sparsity_pattern,
                        method=method, vectorized=False,
                        first_step=1e-8, dense_output=True,
                        mass=Mi, bPrint=bPrint)  
    print('Reference solution CV computed')      
    print('  --> "{}"'.format(ref_sol.message))
    if not ref_sol.success:
        raise Exception('Radau reference CV solution failed')
    
    return ref_sol

# Function to evaluate errors 
def get_errors(nt_vec, order_vec, md_sols, ref_sol):
    err = np.zeros((order_vec.size, nt_vec.size))
    bErrL2 = True
    # dx = options_electrolyte['mesh']['cellSize'][0]
    ref_sol_y= ref_sol.y 
    for i, current_nt in enumerate(nt_vec) :   
        for j, current_order in enumerate(order_vec):
            md_sol_y = md_sols[j][i].y
        
            if (np.any(np.isnan(md_sol_y))):
                errk = np.nan 
            else:
                erri = abs(md_sol_y[:,-1]- ref_sol_y[:,-1])
                if not bErrL2:
                    nrmlz = np.sum(abs(ref_sol_y[:,-1]))
                    if nrmlz < 1e-15:
                        nrmlz += 1e-15
                    errk = np.sum(erri)/nrmlz
                else:
                    nrmlz = np.sqrt(np.sum(abs(ref_sol_y[:,-1])**2))
                    if nrmlz < 1e-15:
                        nrmlz += 1e-15
                    errk = np.sqrt( np.sum( erri**2))/nrmlz
            err[j, i] = errk 
    return err

# Functions to get valid (non NaN) error plots
def get_plt_curves(nt_vec, dt_vec, order_vec, err, qsList=None):
    nSm = order_vec.size
    
    if qsList is None:
        qsList = {}
        for order in order_vec:
            qsList[order] = 0

    qs = np.zeros(nSm, dtype=int)
         
    nt_plt_vec = []
    dt_plt_vec = []
    err_plt_vec = []
    nt_order_vec = []
    dt_order_vec = []
    th_curve_vec = []
    for k, current_order in enumerate(order_vec):
        err_plt = err[k, :]
        nt_plt = nt_vec
        dt_plt = dt_vec
        idx_nan = []
        for i in range(len(err_plt)):
            
            if np.isnan(err_plt[i]):
                idx_nan.append(i)
                qs[k] += 1
                
        if qsList is not None:
            qs[k] = qsList[current_order]
                
        # print(idx_nan)
        if (len(idx_nan) == len(err_plt)): continue

        err_pltk = np.delete(err_plt, idx_nan)
        nt_pltk = np.delete(nt_plt, idx_nan)
        dt_pltk = np.delete(dt_plt, idx_nan)
        
        nt_plt_vec.append(nt_pltk)
        dt_plt_vec.append(dt_pltk)
        err_plt_vec.append(err_pltk)
        
        coeff = (err_plt[qs[k]]/(dt_plt[qs[k]]**current_order))
        dtmf = np.log10(min(nt_plt[qs[k]:]))
        dtm0 = .9*dtmf
        # nms =  nSm-qs[k]
        nms =  nt_vec.size - qs[k]
        # nt_longk = np.r_[np.logspace(dtm0, dtmf, num=nms), nt_plt]
        nt_longk = np.r_[np.logspace(dtm0, dtmf, num=nms), nt_plt[qs[k]:]]
        dt_longk = (nt_pltk[0]-1)*dt_pltk[0]/(nt_longk-1)
        
        nt_order_vec.append(nt_longk)
        dt_order_vec.append(dt_longk)
        th_curve_vec.append(coeff*dt_longk**current_order)
            
    return nt_plt_vec, dt_plt_vec, err_plt_vec, nt_order_vec, dt_order_vec, th_curve_vec
