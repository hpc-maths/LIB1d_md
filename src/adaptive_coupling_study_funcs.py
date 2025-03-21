# -*- coding: utf-8 -*-
# """
# Created on Fri Feb 17 10:08:45 2023

# # -*- coding: utf-8 -*-

# @author: ali.asad
# """
#############IMPORTING MODULES###############
import numpy as np

# customs modules
from src.lib.reader import  reader as params
from src.lib.model import setup_LIB as setup 

from src.lib.model import _1d_lib_dfv_model as coupled_model



from scipy.integrate import solve_ivp
from src.lib.solver.radau_dae import RadauDAE
import scipy.sparse as sparse

from tqdm import tqdm

# Function to perform initial transient simulation using BE method
def initial_transient_sim(
                        tini,
                        tend,
                        y0,
                        nt=1,
                        options_electrolyte=None,
                        options_cathode=None
):
    
    # tols for Newton
    atol = 1e-10 
    rtol = 1e-9
    bPrint=False
    
    assert options_cathode['sim_type']=="monolithic"
    assert options_electrolyte['sim_type']=="monolithic"
    
    Mi = np.diag(np.r_[options_electrolyte['mass'], options_cathode['mass']])
    
    from src.lib.solver import be_dae_solver as be_solver
    out_ini = be_solver.integrate_dae(mass = Mi,
                                    tini = tini,
                                    tend = tend,
                                    nt = nt,
                                    yini = y0,
                                    fcn = lambda x,t: coupled_model.func_U(x, t, options_electrolyte, options_cathode),
                                    options_electrolyte = options_electrolyte,
                                    options_cathode = options_cathode,
                                    rtol=rtol, atol=atol,
                                    verbose = False,
                                    verbose_freq = 1) 
    return out_ini

# Function to perform multi-domain simulations
def perform_adaptive_md_sim(y0_global, t_span,
                            order=None,
                            dt_rtol=1e-6,
                            bExplicitCoupling=True,
                            NITER_MAX=100,
                            adaptive_subsolves=True,
                            options_electrolyte=None,
                            options_cathode=None,
                            bMDSim_v1=False,
                            md_sim_logger=100,
                            coupler_logger=100):
    
    if (options_electrolyte is None):
        raise Exception('options_electrolyte not provided')
    if (options_cathode is None):
        raise Exception('options_cathode not provided')
    
    if order is None:
        raise Exception ('order must be provided')
        
    WR_tol = dt_rtol/5.
    subsolve_tol = WR_tol/5.
    getCV_tol = subsolve_tol/5.
    
    if not bMDSim_v1:
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
            md_sim.raise_error_on_non_convergence = True 
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
            md_sim.raise_error_on_non_convergence = True
            
    y0 = y0_global
    reset_predictors = True
    
    first_step = 1e-6
    max_step = abs(t_span[-1] - t_span[0])/6.
    
    try :
        dt_atol=dt_rtol/10
        md_sim.embedded_method=True
        out = md_sim.adaptive_integration(y0=y0, t_span=t_span,
                                        rtol=dt_rtol, atol=dt_atol,
                                        adaptive_order=False,
                                        reset_predictors=reset_predictors,
                                        first_step=first_step,
                                        max_step=max_step)         
    except Exception as e:
        raise e
        print('Random exception caught \n', e)   
        out = OdeResult()
        out.success = False
        out.message = f"Rejected Multi-Domain Simulation with Early exit due to {e}"
        out.y = np.nan*np.ones(shape=(np.array(y0_global).T.shape)) ## ATTENTION we use this hack only for conv. study
            
    return out


def adaptive_md_study_loop(order_vec,
                        tmd_start,
                        tmd_end,
                        y0_global,
                        dt_rtol,
                        bExplicitCoupling=True,
                        options_electrolyte=None,
                        options_cathode=None,
                        bMDSim_v1=False,
                        nparallel=0):

    if (options_electrolyte is None):
        raise Exception('options_electrolyte not provided')
    if (options_cathode is None):
        raise Exception('options_cathode not provided')
    
    t_vec = np.array([tmd_start, tmd_end])
    
    # y0_global, t_vec,
    # order=None,
    # dt_rtol=1e-6,
    # bExplicitCoupling=True,
    # NITER_MAX=100,
    # adaptive_subsolves=True,
    # options_electrolyte=None,
    # options_cathode=None,
    
    if nparallel==0: # sequential
        sim_md_sols = []
        for i, current_order in enumerate(order_vec):
            out_md_sim = perform_adaptive_md_sim(y0_global=y0_global,
                                               t_span=t_vec,
                                               order=current_order,
                                               dt_rtol=dt_rtol,
                                               bExplicitCoupling=bExplicitCoupling,
                                               options_electrolyte=options_electrolyte,
                                               options_cathode=options_cathode,
                                               bMDSim_v1=bMDSim_v1, # coupling code version for paper
                                               NITER_MAX=100,
                                               md_sim_logger=100)
    
            sim_md_sols.append(out_md_sim)
    else: # parallel
      
        sim_md_sols = []
        
        from joblib import Parallel, delayed, parallel_config
        from itertools import product
        
        def parfun(current_order):
            return perform_adaptive_md_sim(y0_global=y0_global,
                                        t_span=t_vec,
                                        order=current_order,
                                        dt_rtol=dt_rtol,
                                        bExplicitCoupling=bExplicitCoupling,
                                        options_electrolyte=options_electrolyte,
                                        options_cathode=options_cathode, 
                                        bMDSim_v1=bMDSim_v1, # coupling code version for paper
                                        NITER_MAX=100,
                                        md_sim_logger=100)
                
        data = list(order_vec)
        with parallel_config(backend="loky", inner_max_num_threads=2):
            pool = Parallel(n_jobs=nparallel, verbose=1000)
            results = pool(delayed(parfun)(current_order) for current_order in data)
        
        sim_md_sols = results
          
      
    print('------------------------------\nstudy loop ended')
    if True:
        for i, current_order in enumerate(order_vec):
            print(f"p={current_order-1} : {sim_md_sols[i].message}")
    
    return sim_md_sols



# Function to run work-precision study loop
def work_precision_loop(dt_rtol_vec,
                        order_vec,
                        t_md_start,
                        t_md_end,
                        y0_global,
                        bExplicitCoupling=False,
                        options_electrolyte=None,
                        options_cathode=None,
                        nparallel=0):
    
    if (options_electrolyte is None):
        raise Exception('options_electrolyte not provided')
    if (options_cathode is None):
        raise Exception('options_cathode not provided')
    
    if bExplicitCoupling:
        print(f"adaptive explicit coupling for orders, p={order_vec-1}")
    else:
        print(f"adaptive implicit coupling for orders, p={order_vec-1}")

    t_vec = np.array([t_md_start, t_md_end])                
    if nparallel==0:        
        md_sim_sols = [[None for i in dt_rtol_vec] for k in order_vec]
        for i, current_dt_rtol in enumerate(tqdm(dt_rtol_vec)):
            for j, current_order in enumerate(order_vec):
                # if current_order == 1:
                #     dt_rtol = min(current_dt_rtol*1e4, max(dt_rtol_vec)*1e1/(5-i)) 
                # elif current_order == 2:
                #     dt_rtol = min(current_dt_rtol*1e2, max(dt_rtol_vec)*1e1/(5-i))
                # else:
                #     dt_rtol = current_dt_rtol        
                dt_rtol = current_dt_rtol        
                out_md_sim = perform_adaptive_md_sim(y0_global=y0_global,
                                                    t_span=t_vec,
                                                    order=current_order,
                                                    dt_rtol=dt_rtol,
                                                    bExplicitCoupling=bExplicitCoupling,
                                                    options_electrolyte=options_electrolyte,
                                                    options_cathode=options_cathode,
                                                    bMDSim_v1=True,
                                                    NITER_MAX=100,
                                                    md_sim_logger=100)
    
            print("pmax =", current_order-1,
                  " rtol =", f"{dt_rtol:.0E}",
                  " : simulation", out_md_sim.message if (out_md_sim is not None) else 'Failure')
            md_sim_sols[j].append(out_md_sim)
    else:
        md_sim_sols = [[None for i in dt_rtol_vec] for k in order_vec]
        from joblib import Parallel, delayed, parallel_config
        from itertools import product
        def parfun(current_dt_rtol, current_order):
            # i = dt_rtol_vec.tolist().index(current_dt_rtol)
            # if current_order == 1:
            #     dt_rtol = min(current_dt_rtol*1e4, max(dt_rtol_vec)*1e1/(5-i)) 
            # elif current_order == 2:
            #     dt_rtol = min(current_dt_rtol*1e2, max(dt_rtol_vec)*1e1/(5-i))
            # else:
            #     dt_rtol = current_dt_rtol
            dt_rtol = current_dt_rtol
            return perform_adaptive_md_sim(y0_global=y0_global,
                                            t_span=t_vec,
                                            order=current_order,
                                            dt_rtol=dt_rtol,
                                            bExplicitCoupling=bExplicitCoupling,
                                            options_electrolyte=options_electrolyte,
                                            options_cathode=options_cathode,
                                            bMDSim_v1=True,
                                            NITER_MAX=100,
                                            md_sim_logger=100)
                
        data = list(product(dt_rtol_vec, order_vec))
        with parallel_config(backend="loky", inner_max_num_threads=2):
            pool = Parallel(n_jobs=nparallel, verbose=1000)
            results = pool(delayed(parfun)(current_dt_rtol, current_order) for current_dt_rtol, current_order in data)

        for it, ((current_dt_rtol, current_order), out) in enumerate(zip(list(data),results)):
            i = dt_rtol_vec.tolist().index(current_dt_rtol)
            j = order_vec.tolist().index(current_order)
            md_sim_sols[j][i] = out
            
            # if current_order == 1:
            #     dt_rtol = min(current_dt_rtol*1e4, max(dt_rtol_vec)*1e1/(5-i)) 
            # elif current_order == 2:
            #     dt_rtol = min(current_dt_rtol*1e2, max(dt_rtol_vec)*1e1/(5-i))
            # else:
            #     dt_rtol = current_dt_rtol
            
            dt_rtol = current_dt_rtol
            print("pmax =", current_order-1,
                  " rtol =", f"{dt_rtol:.0E}",
                  " : simulation", out.message if (out is not None) else 'Failure')
            

    return md_sim_sols


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
def get_errors(dt_rtol_vec, order_vec, md_sols, ref_sol):
    err = np.zeros((order_vec.size, dt_rtol_vec.size))
    bErrL2 = True
    # dx = options_electrolyte['mesh']['cellSize'][0]
    ref_sol_y= ref_sol.y 
    for i, current_dt_rtol in enumerate(dt_rtol_vec) :   
        for j, current_order in enumerate(order_vec):
            if md_sols[j][i] is None:
                md_sol_y = np.nan
            else:
                md_sol_y = md_sols[j][i].y
                
            if (np.any(np.isnan(md_sol_y))):
                errk = np.nan 
            else:
                erri = abs(md_sol_y[:,-1]- ref_sol_y[:,-1])
                
                errki = erri/(1e-15 + abs(ref_sol_y[:,-1]))
                errk = np.linalg.norm(errki) #/ np.sqrt(errki.size)
        
            err[j, i] = errk 
    return err

def get_CPUtimes(dt_rtol_vec, order_vec, md_sols):
    cpu_times = np.zeros((order_vec.size, dt_rtol_vec.size))
    for i, current_dt_rtol in enumerate(dt_rtol_vec) :   
        for j, current_order in enumerate(order_vec):
            cpu_times[j, i] = md_sols[j][i].CPUtime
    
    return cpu_times

def get_work_precision_curves(order_vec,
                              CPUtimes_vec,
                              errors_vec):
    
    error_plt_list = []
    CPUTime_plt_list = []
            
    for k, current_order in enumerate(order_vec):
        err_plt = errors_vec[k, :]
        cpu_plt = CPUtimes_vec[k, :]
        
        idx_nan = []
        for i in range(len(err_plt)):
            
            if np.isnan(err_plt[i]):
                idx_nan.append(i)
                
        err_pltk = np.delete(err_plt, idx_nan)
        cpu_pltk = np.delete(cpu_plt, idx_nan)
        
        error_plt_list.append(err_pltk)
        CPUTime_plt_list.append(cpu_pltk)
        
    return error_plt_list, CPUTime_plt_list