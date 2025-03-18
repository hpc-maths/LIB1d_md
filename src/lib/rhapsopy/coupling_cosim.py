#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:03:05 2021

Test two slabs coupled at an interface with co-simulation techniques or with a monolithic approach

@author: laurent
"""
import numpy as np
np.seterr(divide="raise")
np.set_printoptions(precision=13)
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.optimize import OptimizeResult as OdeResult
import time as pytime
import logging
logging.raiseExceptions = True

from src.lib.rhapsopy.prediction_cosim import predicteur

# logging levels
FIXEDPOINT = 11
ORCHESTRATOR = 31
INTEGRATION = 31 # start and end of integration
INTEGRATION_DETAIL = 21 # integration steps
INTEGRATION_DETAIL2 = 16 # more info on convergence
INTEGRATION_DETAIL3 = 11 # more info on convergence
INTEGRATION_DETAIL4 = 6 # detailed information about everything
# TODO: different loggers or filters for each part (fixed-point, integration)

# ERROR CODES
WRNONCONVERGENCE  = 1 # main method failed
WRNONCONVERGENCE2 = 2 # embedded method failed
OTHEREXCEPTION = 5    # other exception during step
ERRORTOOHIGH = 6      # error estimate too large
ACCEPTED = 0          # step is accepted

# Cuisine interne
TOLERANCE_FACTOR_DTOPT = 1.1
MAXRELSTEP = 2.0 # maximum increase factor for the time step
MINRELSTEP = 0.1 # minimum ...
SAFETY_FACTOR = 0.9

ncalls=None
others=None


class WRnonConvergence(Exception):
    """ Exception class used in cased of convergence failure for implicit cosimulations """
    pass

class BaseCoupler():
    """ Model for the coupler class that handles the subsystem integration """
    def __init__(self, options1, options2, coupling_modes):
        self.nCouplingVars = 5 # number of coupling variables
        self.predictors = None # Dictionnary form of the polynomial predictors used by the co-simulation class, for better handling

    # def _spreadPredictors(self, pred_list):
    #     raise NotImplementedError()

    # def _feedOptions(self, t, last_outs):
    #     raise NotImplementedError()

    def getCouplingVars(self,t,y):
        raise NotImplementedError()

    def integrateSingleSubsystem(self, isolv, t0, y0, dt, preds, last_outs, rtol=None, bDebug=False):
        raise NotImplementedError()

    #def getErrors(self, y, yref, t, i_iter, atol, rtol, bPrint):
    #    raise NotImplementedError()

#%% Acceleration methods
class IQNSolver():
  def __init__(self):
    self.W = []
    self.V = []
    self.J = None
    self.max_used_steps=None

  def computeError(self, x1, x2, rtol):
    return  np.linalg.norm( (x1-x2) / ( rtol + rtol*abs(x2) ) )   /   np.sqrt(x1.size)
  
  def solve(self, fun, ftol, rtol, nitemax, x0, omega0=0.5):
    x      = [x0.copy()]
    xtilde = [fun(x0)]
    R      = [xtilde[0] - x[0]]
    
    # 1st iteration
    it = 0
    x.append( x0 + omega0*R[-1] )
    error = self.computeError(x[-1], x[-2], rtol=rtol)
    print(f'\tit={it}, err={error:.2e}')
    
    bConverged = False
    # ak denotes a^k (superscript), while a_k denotes an indice
    while not bConverged:
      it+=1
      xk  = x[-1]
      xtk = fun(xk)
      xtilde.append( xtk )
      
      # early error check
      error = self.computeError(xtk, xk, rtol=rtol)
      if error < 1.:
        bConverged=True
        break

      # form residual vector
      Rk = xtk - xk
      R.append( Rk )        
      if len(R)>4:
        R_k      =      np.array(R[-4:]).T
        xt_array = np.array(xtilde[-4:]).T
        
        V_k = np.diff(R_k,  axis=1)
        W_k = np.diff(xt_array, axis=1)
        # if V_k.ndim==1: # only 1 coupling variable
        #   V_k = V_k[np.newaxis,:]
        #   W_k = W_k[np.newaxis,:]
          
        if 0:
          # QR decomposition of V_k
          Q,U = scipy.linalg.qr(V_k, overwrite_a=False, lwork=None, mode='full',
                          pivoting=False, check_finite=True)
          # solve
          print('shape(V_k)=', V_k.shape)
          print('shape(Q)=',   Q.shape)
          print('shape(U)=',   U.shape)
          alpha = scipy.linalg.solve(a=U, b=-Q.T @ R_k, check_finite=True)
          dxtk = W_k @ alpha
        else:
          # direct pseudo-inverse, solve V_k alpha = - R_k
          alpha = - np.linalg.pinv( V_k ) @ R_k
          dxtk = W_k @ alpha

        x_kp1 = xtk + dxtk
        x.append( x_kp1.copy() )
        
      else: # underrelaxed fixed-point
        # x.append( x[-1] + omega0*R[-1] )
        x.append( xk    + omega0*(xtk - xk) )
        
      error = self.computeError(x[-1], x[-2], rtol=rtol)
      print(f'\tit={it}, err={error:.2e}')
      if error<1e-15: # issue at first iteration ?
        print('weird')
        continue
      if error < 1.:
        bConverged = True
      if not bConverged and (it>nitemax):
        raise Exception('IQN did not converge')
    return x[-1]

  
class AitkenUnderrelaxation():
  def __init__(self):
    pass
  
  def computeError(self, x1, x2, rtol):
    return  np.linalg.norm( (x1-x2) / ( rtol + rtol*abs(x2) ) )   /   np.sqrt(x1.size)
  
  def solve(self,fun, ftol, rtol, nitemax, x0, omega0=0.5):
    x      = [x0.copy()]
    xtilde = [fun(x0)]
    omega  = [omega0]
    
    # 1st iteration
    it = 1
    x.append( omega[-1]*xtilde[-1]+(1-omega[-1])*x[-1] )
    xtilde.append( fun(x[-1]) )
    error = self.computeError(x[-1], x[-2], rtol=rtol)
    print(f'\tit={it}, omeg={omega[-1]:.2e}, err={error:.2e}')
    
    bConverged = False
    
    while not bConverged:
      it+=1
      omega_km1  = omega[-1]
      x_km1      = x[-2]
      xtilde_km1 = xtilde[-2]
      x_k          = x[-1]
      xtilde_k     = xtilde[-1]
      
      R_k = xtilde_k - x_k
      R_km1 = xtilde_km1 - x_km1
      
      omega_k = - omega_km1 * R_km1.T.dot(R_k-R_km1) / np.linalg.norm(R_k - R_km1)**2
      x_kp1 = omega_k * xtilde_k + (1-omega_k)*x_k
      x.append( x_kp1.copy() )
      omega.append(omega_k)
      
      error = self.computeError(x[-1], x[-2], rtol=rtol)
      print(f'\tit={it}, omeg={omega[-1]:.2e}, err={error:.2e}')
      if error < 1.:
        bConverged = True
        break

      xtilde.append( fun(x_kp1) )
      if not bConverged and (it>nitemax):
        raise Exception('Aitken underrelaxation did not converge')
    return x[-1]
      
#%% Co-simulation wrapper
class Orchestrator:
  """ This class sets up and performs the coupled integration of the multiple subsystems,
  using a co-simulation approach to enable high-order time accuracy. """

  def __init__(self, coupler, NMAX):
    """ Initialize the coupled integration """
    # create logger
    self.logger = logging.getLogger("cosim.coupling")
    self.logger.handlers = [] # drop previous handlers
    self.logger.setLevel(100) # no logging by default

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(1) # no filtering of logs
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(formatter)
    self.logger.addHandler(ch)
    self.logger.log(ORCHESTRATOR, 'Setting up orchestrator object')


    self.coupler = coupler
    self.subsystemsIntegratedInSingleCall = False

    # If True, the function "integrateAllSubsystems" of the coupler object
    # is called to integrate all subsystems, as for a parallel Jacobi-type simulation.
    # Otherwise, the function "integrateSingleSubsystem" is called to integrate
    # the subsystems one by one.

    self.NMAX = NMAX
    self.logger.log(ORCHESTRATOR, f'maximum prediction order set to {self.NMAX}')
    self.preds = [ predicteur(NMAX=NMAX) for i in range(coupler.nCouplingVars) ]

    # WR iteration
    #  parameters
    self.waveform_tolerance = 1e-12 # tolerance for the termination fo the waveform iterations
    self.gauss_seidel = False # WR iteration scheme
    self.cosim_ordering = list(range(coupler.nSubsystems)) # solver ordering for WR iteration scheme (only meaningful if Gauss-Seidel mode is activated)
    self.raise_error_on_non_convergence = True # raise an error if the WR iteration do not converge
    self.NITER_MAX = 1000 # maximum number of WR iterations per step

    # convergence acceleration
    self.interfaceSolver = 'fixed-point' # use a Newton method TODO: implement properly
    self.logger.log(ORCHESTRATOR, f'Iteration method is {self.interfaceSolver}')
    self.checkConvergenceRate = True # if True, WR convergence rate is monitored

    # adaptive integration
    self.embedded_method = False # if True, the coupling error is estimated by comparing the coupling variables
     # obtained for different approximation orders (better for implicit coupling)


  def _step_forward(self, t, yn, dt, values_last_iterate, last_outs=None, rtol=None):
    """ Performs one iteration of a co-simulation step (Jacobi or Gauss-Seidel)
          --> computes the value of the overall state vector at time t+dt,
              starting from state y at time t. """
    self.logger.log(INTEGRATION_DETAIL3, 'performing single step iteration')
    # Update the value of the last point (remember that we are now working in interpolation mode based on the previous
    # iteration of the current time step)
    self._updatePredictors(t=t+dt, coupling_vars=values_last_iterate)

    # TODO: do not carry rtol here, but let the user specify it to the coupler object directly ?
    if self.subsystemsIntegratedInSingleCall: # all subsystems are integrated in a Jacobi manner at once
        self.logger.log(INTEGRATION_DETAIL4, ' subsystems will be integrated in a single coupler call')
        outs = self.coupler.integrateAllSubsystems(t0=t, y0=yn, dt=dt, preds=self.preds,
                                                last_outs=last_outs, rtol=rtol)
        newy = [o.y[:,-1] for o in outs]
    else: # Jacobi or Gauss-Seidel approach, one subsystem at a time
        outs = [None for i in range(len(self.cosim_ordering))]
        newy = [None for i in range(len(self.cosim_ordering))]
        for it,isolv in enumerate(self.cosim_ordering):
            self.logger.log(INTEGRATION_DETAIL4, f' integrating subsystem {isolv}')
            outs[isolv] = self.coupler.integrateSingleSubsystem(isolv=isolv, t0=t, y0=yn, dt=dt, preds=self.preds,
                                                           last_outs=last_outs, rtol=rtol)
            newy[isolv] = outs[isolv].y[:,-1]

            if self.gauss_seidel and ( it < 1 ): # do not update after the last solve
              raise Exception('TODO: investigate the O(dt) error...')
              # new_last_outs = last_outs
              ## get each subsystem's state vector
              # nx1 = self.options1['y0'].size
              # y1 = yn[:nx1]
              # y2 = yn[nx1:]
              # update BCs          current_y=newy[isolv]
              # if isolv==0:
              #   other_y=last_ynp1[nx1:]
              #   y_cat = np.hstack( (current_y,other_y) ) # bof at first call
              # else:
              #   other_y=last_ynp1[:nx1]
              #   y_cat = np.hstack( (other_y,current_y) )
              # # this introduces an O(dt) error, since we use the other state at tn, not tn+dt...
              # # TODO: the updated values should be computed from the new state (left or right) we just computed,
              # # and from the prediction of the other one !
              # y_cat = np.nan*yn
              # if isolv==0:
              #   y_cat[nx1] = self.predictors['right']['Tbnd'].eval_single(t+dt)
              #   y_cat[nx1-1] = current_y[-1]
              # else:
              #   y_cat[nx1-1] = self.predictors['left']['Tbnd'].eval_single(t+dt)
              #   y_cat[nx1] = current_y[0]
              # self._updatePredictors(t=t+dt, coupling_vars=self._getCouplingVars(t=t+dt, y=last_ynp1))
              # new_last_outs[isolv] = current_out
              # # for i in range(2): # for each subsystem
              # #   if i!=isolv:
              # #     self.coupler._feedOptions(t=t, i=i, last_outs=new_last_outs)

    self.logger.log(INTEGRATION_DETAIL4, '     subsystem substeps: {}'.format([o.t.size-1 for o in outs]))
    # print('subsystem substeps: {}'.format([o.t.size-1 for o in outs]))
    self.logger.log(INTEGRATION_DETAIL4, 'step iteration performed')

    # construct overall time derivatives
    return np.hstack(newy), outs


  def perform_step(self, y0, t, dt, atol_iter=None, rtol_iter=None, bDebug=False, bPrint=True, rtol=None):
        """ Perform the WR iteration process for a single time step """
        self.logger.log(INTEGRATION_DETAIL3, 'performing step')
        bConverged = False
        tn = t # current starting time
        tnp1 = t+dt # next coupling time

        ynp1_kp1 = y0 # TODO: better guess
        outs_kp1 = None # backup of the previous dense outputs

        if dt<1e-12:
            raise Exception('stop dt is too low')
        assert not (self.preds[0].x is None), 'predictors have not been initialised !!!'


        if atol_iter is None:
          atol_iter = self.waveform_tolerance
        if rtol_iter is None:
          rtol_iter = self.waveform_tolerance

        pred_coupling_vars = np.array([p.eval_single(tnp1, allow_outside=True) for p in self.preds])[:,0]

        ## Since we are at the first iteration, we only have data up to time t
        # and extrapolate on [t, t+dt]
        # To do so:
        # - we first register the solution at time t once at the first iteration
        #   of the new time step (which starts from t and goes to t+dt)
        # - then we evaluate the extrapolation at t+dt and register a fake point
        #   at t+dt with these values (this drops the oldest interpolation point,
        #   but does not affect the prediction polynomial)
        # - then, for the other iterations of the current time step, we simply
        #   update these last data points with the current iteration's values,
        #   so that we work in interpolation mode instead !
        # 1 - We append these points
        if self.embedded_method:
          pass  # we do nothing here, since the embedded approach already provides the proper perdictor setup
        else:
          if abs((self.preds[0].x[0] - tnp1)/dt) < 1e-2:
                raise Exception('Time tnp1 seems to have already been registered in the predictors...')
          else:
            oldNs = [p.N for p in self.preds]
            self._advancePredictors(t=tnp1, coupling_vars=pred_coupling_vars)
            # discard the oldest point so that the polynomial is unaffected (in particular its order...)
            # however, we must set N back to its correct value after the WR iterations are done and the step is accepted
            for oldN,pred in zip(oldNs,self.preds):
                pred.setCurrentN(oldN)
            # bPredNmaxedOut = [p.N<p.NMAX for p in self.preds]
            # for (bNeedsCorrection, pred) in zip(bPredNmaxedOut, self.preds):
            #     if bNeedsCorrection:
            #         pred.decreaseN() # to avoid using the oldest point, which is now "redundant"

        ### WR iterations to converge the present coupling step
        iterated_values = np.zeros((self.NITER_MAX+1,len(self.preds)))
        iterated_values[0,:] = pred_coupling_vars
        nerrors=2
        errors = np.zeros((self.NITER_MAX,nerrors))
        coupling_vars_kp1 = np.array([p.eval_single(tnp1)[0] for p in self.preds])

        # TODO: automatic error control for all predicted variables + user-defined variables (which are not predicted)
        # TODO: complex step once to identify the dependence (boolean) of the errors on the predictors, so that the error orders are roughly known ?
        global ncalls
        global others
        ncalls=0
        others={}
        def fixedPointFunction(coupling_vars_k, outs_kp=None, full_output=False):
            """ Fixed point function.
                Inputs:
                    - coupling_vars_k:  array
                        guess of the coupling variables at time t_{n+1}
                    - outs_kp: list or tuple of OdeSolution objects
                        solution structure (including dense outputs) of the
                        previous iteration
                    - full_output: boolean
                        if True, additional outputs are given
                Output:
                    - coupling_vars_kp1: array
                        the updated coupling variables at t_{n+1}, obtained
                        after integrating the subsystems separately from time
                        t_{n} to t_{n+1} and recomputing the coupling variables
                        based on the new final states
                    - ynp1_kp1: array
                        new overall state vector
                    - outs_kp1: list or tuple of OdeSolution objects
                        solution structure of the new subsystems solutions
            """
            # TODO: it would be more sensible for large-scale applications to
            # let each solver object store (if asked by the orchestrator) its data
            # and only transmit to the coupler the  coupling variables.
            # The coupler would ask each solver to update, store, and validate
            # (after step acceptance) its state vector.
            # print(' --> coupling_vars_k=',coupling_vars_k[0])
            global ncalls
            global others
            ncalls+=1
            ynp1_kp1, outs_kp1 = self._step_forward(yn=y0, t=tn, dt=dt,
                                           values_last_iterate=coupling_vars_k,
                                           last_outs=outs_kp, rtol=rtol)
            coupling_vars_kp1 = self._getCouplingVars(t=tnp1, y=ynp1_kp1)
            others["ynp1_kp1"] = ynp1_kp1
            others["outs_kp1"] = outs_kp1
            if full_output:
                return coupling_vars_kp1, ynp1_kp1, outs_kp1
            else:
                return coupling_vars_kp1

        def residfun(coupling_vars):
            """ Residual function for a Newton approach """
            return coupling_vars - fixedPointFunction(coupling_vars)

        # Accelerated fixed-point #scipy.optimize.fixed_point
        if self.interfaceSolver=='fixed-point' or self.interfaceSolver=='aitken':
            if self.interfaceSolver=='fixed-point':
                method='iteration'
            else:
                method='del2'
            coupling_vars_kp1, i_iter = self.fixed_point(func=fixedPointFunction,
                                    x0=coupling_vars_kp1.copy(), args=(), xtol=rtol_iter,
                                    maxiter=self.NITER_MAX, method=method)
            ynp1_kp1 = others["ynp1_kp1"]
            outs_kp1 = others["outs_kp1"]
        elif self.interfaceSolver=='scipy_newton':
            coupling_vars_kp1 = scipy.optimize.newton(func=lambda x: x-fixedPointFunction(x),
                                              x0=coupling_vars_kp1.copy(), fprime=None,
                                              args=(), tol=1e-15, maxiter=self.NITER_MAX,
                                              fprime2=None, x1=None, rtol=rtol_iter,
                                              full_output=False, disp=True)
            ynp1_kp1 = others["ynp1_kp1"]
            outs_kp1 = others["outs_kp1"]
            
        elif self.interfaceSolver=='scipy_anderson':
            coupling_vars_kp1 = scipy.optimize.anderson(F=lambda x: x-fixedPointFunction(x),
                                                        xin=coupling_vars_kp1.copy(),
                                                        alpha=None, w0=0.01, M=5, verbose=False,
                                                        maxiter=self.NITER_MAX, f_tol=1e-12, f_rtol=None,
                                                        x_tol=atol_iter, x_rtol=rtol_iter,
                                                        tol_norm=None, line_search='armijo')
            ynp1_kp1 = others["ynp1_kp1"]
            outs_kp1 = others["outs_kp1"]
    
        elif self.interfaceSolver=='AitkenUnderrelaxation':
            solver = AitkenUnderrelaxation()
            coupling_vars_kp1 = solver.solve(fun=fixedPointFunction,
                                             x0=coupling_vars_kp1.copy(),
                                             ftol=1e-15, rtol=rtol_iter,
                                             nitemax=self.NITER_MAX)
            ynp1_kp1 = others["ynp1_kp1"]
            outs_kp1 = others["outs_kp1"]

        elif self.interfaceSolver=='IQN':
            solver = IQNSolver()
            coupling_vars_kp1 = solver.solve(fun=lambda x: x-fixedPointFunction(x),
                                             x0=coupling_vars_kp1.copy(),
                                             ftol=1e-15, rtol=rtol_iter,
                                             nitemax=self.NITER_MAX)
            ynp1_kp1 = others["ynp1_kp1"]
            outs_kp1 = others["outs_kp1"]

        elif self.interfaceSolver=="explicit":  # no iteration --> explicit cosim
            # TODO: improve
            # TODO: y should no be considered in the orchestrator
            # at most, the orchestrator should only be able to ask the coupler object to store y if necessary
            for i in range(self.NITER_MAX): # possibility to do more than 1 iteration, even though this should not be used
                coupling_vars_kp1, ynp1_kp1, outs_kp1 = fixedPointFunction(coupling_vars_kp1,
                                                               outs_kp=None, full_output=True)
        else:
            raise Exception('Interface solver {} unknown'.format(self.interfaceSolver))


        i_iter = ncalls
        # coupling_vars_kp1, ynp1_kp1, outs_kp1 = fixedPointFunction(coupling_vars_kp1,
                                                       # outs_kp=None, full_output=True)
        bConverged = True
        self.logger.log(INTEGRATION_DETAIL4, 'step performed')
        return bConverged, i_iter, ynp1_kp1

  def _getCouplingVars(self,t,y):
    """ Call each system's object to gather the required coupling variables """
    self.logger.log(INTEGRATION_DETAIL4, 'getting coupling variables')
    return self.coupler.getCouplingVars(t,y)
  
  def _getPredOrders(self):
      """ Return current prediction orders """
      return np.array([p.N for p in self.preds])

  def _updatePredictors(self, t, coupling_vars):
    """ Update the predictors by refreshing the coupling variables at the last sampling point """
    self.logger.log(INTEGRATION_DETAIL4, 'updating predictors')
    for i, var in enumerate(coupling_vars):
        self.preds[i].replaceLastPoint_single(x=t, y=var)

  def _advancePredictors(self, t, coupling_vars):
    """ Add a new sampling point to the predictors """
    self.logger.log(INTEGRATION_DETAIL4, 'advancing predictors with new coupling variables')
    for i, var in enumerate(coupling_vars):
        self.preds[i].appendData_single(x=t, y=var)

  def _backupPredictors(self):
    """ Backup the current predictors, so as to be able to restore them in case of step failure,
    avoiding the possibility of conveying errors due to sampling point modifications """
    self.logger.log(INTEGRATION_DETAIL4, 'backing up predictors')
    return [p.clone() for p in self.preds]

  def _restorePredictors(self, preds):
    """ Restore the predictors to a previous backup """
    self.logger.log(INTEGRATION_DETAIL4, 'restoring predictors')
    self.preds = [p.clone() for p in preds]

  def basic_integration(self, y0, t_vec, reset_predictors=True, nDebugAfterNsteps=np.inf):
    """ Basic integration with prescribed time steps (t_vec) with waveform relaxation for each step """
    self.logger.log(INTEGRATION, 'Starting coupled simulation with prescribed time steps')
    self.logger.log(INTEGRATION, f' iteration algorithm is {self.interfaceSolver}')
    tstart = pytime.time()
    nt = t_vec.size

    iter_hist = [] # WR iterations per step
    t0 = t_vec[0]
    
    if reset_predictors:
      # reset predictors to ensure they do not contain previously acquired data
      # else, assume they have been properly setup
      for p in self.preds:
        p.reset()
      assert self.preds[0].x is None, 'Predictors seem to have been initialised already'
      # add the starting point as initial data for the predictors
      
      ini_coupling_vars = self._getCouplingVars(t=t0, y=y0)
      self._advancePredictors(t=t0, coupling_vars=ini_coupling_vars)

    yhist = [y0]
    couplingvar_hist = [ np.hstack([p.eval_single(t0) for p in self.preds]) ]
    ##### Temporal loop #####
    for i in range(1,nt): # go from tn to tn+1
      bDebug = nDebugAfterNsteps<i # if True, make lots of plot and quit
      tn   = t_vec[i-1]
      tnp1 = t_vec[i]
      dt   = tnp1 -  tn
      self.logger.log(INTEGRATION, f"t={tn:.3e} s, dt={dt:.2e} s (step {i}/{nt-1})")

      backup_preds = self._backupPredictors()

      # bConverged, niter, ysol = self.perform_step(y0=yhist[i-1], t=tn, dt=dt, bDebug=bDebug)
      
      try:
          bConverged, niter, ysol = self.perform_step(y0=yhist[i-1], t=tn, dt=dt, bDebug=bDebug)
      except Exception as e:
          # import pdb; pdb.set_trace()
          print("\nSimulation exited with:\n", e)
          tend = pytime.time()
          out = OdeResult()
          out.t = t_vec
          out.y = np.nan*np.ones(shape=(np.array(yhist).T.shape))
          out.z = np.array(couplingvar_hist).T
          out.success = True
          out.message = 'Rejected Simulation'
          out.WR_iters = np.array( iter_hist )
          out.nWRtotal = np.sum( out.WR_iters )
          out.CPUtime = tend-tstart
          return out ## ATTENTION we use this hack only for conv. study
      
          # for ivar,var in enumerate(np.array(couplingvar_hist).T):
          #     plt.figure()
          #     plt.plot(t_vec[:i], var)
          #     plt.xlabel('t'); plt.ylabel(f'var {ivar}')
          # import pdb; pdb.set_trace()
          # # try once more in debug mode
          # self._restorePredictors(backup_preds)
          # bConverged, niter, ysol = self.perform_step(y0=yhist[i-1], t=tn, dt=dt, bDebug=bDebug)
          # raise e
          
      if not bConverged:
        self.logger.log(INTEGRATION_DETAIL, 'Loop has not converged after {} iterations'.format(niter))
        if self.raise_error_on_non_convergence:
          raise WRnonConvergence('WR Loop has not converged')
      yhist.append( np.copy(ysol) )
      iter_hist.append(niter)
      couplingvar_hist.append( np.hstack([p.eval_single(tnp1) for p in self.preds]) )

      self.logger.log(INTEGRATION_DETAIL, f'  converged in {niter} iterations')
      # Update predictors
      if 0: # assume everything is well handled :)
          self._updatePredictors(t=tnp1, coupling_vars=self._getCouplingVars(t=tnp1, y=ysol))
          # In this case, the new predictors already have a "fake" interpolation point at tnp1,
          # for which we decreased the number N of sampling points so that the order is unchanged.
          # We now recover the "oldest" data point for the predictors, so that we may
          # increase the order of the prediction !
          for pred in self.preds:
              pred.increaseN()
      else: # safer: restore predictor backups
          self._restorePredictors(backup_preds)
          self._advancePredictors(t=tnp1, coupling_vars=self._getCouplingVars(t=tnp1, y=ysol))

    self.logger.log(INTEGRATION, 'Coupled simulation successfully reached end time')
    tend = pytime.time()
    out = OdeResult()
    out.t = t_vec
    out.y = np.array(yhist).T
    out.z = np.array(couplingvar_hist).T
    out.success = True
    out.message = 'Success'
    out.WR_iters = np.array( iter_hist )
    out.nWRtotal = np.sum( out.WR_iters )
    out.CPUtime = tend-tstart
    return out


  def adaptive_integration(self, y0, t_span, atol, rtol, reset_predictors=True,
                           adaptive_order=False, first_step=1e-6, max_step=np.inf,
                           nPrintLevel=np.inf, bDebug=False):
    """ Adaptive integration with prescribed error tolerances """

    self.logger.log(INTEGRATION, "Adaptive integration begins")

    assert len(t_span)==2
    tstart = pytime.time()

    if reset_predictors:
      # reset predictors
      for p in self.preds:
        p.reset()
      assert self.preds[0].x is None, 'Predictors seem to have been initialised already'

      # add the starting point as initial data for the predictors
      t0 = t_span[0]
      self._advancePredictors(t=t0, coupling_vars=self._getCouplingVars(t=t0, y=y0))
    for p in self.preds:
      p.setTolerances(atol=atol,rtol=rtol)

    ##### Temporal loop #####
    yhist  = [y0]
    thist = [t_span[0]]
    couplingvar_hist = [ np.hstack([p.eval_single(t0) for p in self.preds]) ]
    iter_hist = [] # WR iterations per step
    iter_hist2 = [] # WR iterations per step for the embedded method
    p_hist = [[p.N for p in self.preds]] # prediction orders

    tn = t_span[0]
    dt = first_step
    i=0
    nsteps_total = 0
    nsteps_rejected = 0
    nsteps_accepted = 0
    nsteps_failed = 0
    step_info = []

    while tn<t_span[-1]:
      bAccepted = False
      ntry=1
      i+=1
      consecutive_fails = -1
      self.logger.log(INTEGRATION_DETAIL,        f"\ttn={tn:.3e} s (step {i})")
      initial_extraps = self._backupPredictors()# to keep the original extrapolated values
      #TODO: improve by just keeping the predicted values for each possible order ?
      
      while not bAccepted:
        tnp1 = tn+dt # next coupling time
        self._restorePredictors(preds=initial_extraps)
        if consecutive_fails>2: # lower the order of the predictors
          self.logger.log(INTEGRATION_DETAIL2,'Lowering the prediction order due to repeated failures')
          for p in self.preds:
            p.reportDiscontinuity()
          initial_extraps = self._backupPredictors()

        self.logger.log(INTEGRATION_DETAIL2,f"\t tn={tn:.3e} s, dt={dt:.3e} (try {ntry})")
        bConverged = None
        try:
          # TODO: handle non convergence of the subsystem solves
          bConverged, niter, ysol = self.perform_step(y0=yhist[i-1], t=tn,
                                                      dt=dt, bDebug=bDebug,
                                                      bPrint=(nPrintLevel>2),
                                                      rtol=rtol/20,
                                                      atol_iter=rtol/5,
                                                      rtol_iter=rtol/5)
          if not bConverged:
              self.logger.log(INTEGRATION_DETAIL3,"\t solution did not converge")
              raise WRnonConvergence()
          else:
              self.logger.log(INTEGRATION_DETAIL3,"\t solution converged")

          coupling_vars_np1 = self._getCouplingVars(t=tnp1, y=ysol)
          self._updatePredictors(t=tnp1, coupling_vars=coupling_vars_np1)
          main_preds = self._backupPredictors()

          if self.embedded_method:
              # perform the step anew with a higher-order approximation to use as error estimate
              self.logger.log(INTEGRATION_DETAIL3,"\t computing embedded solution")
              self._restorePredictors(initial_extraps)
              old_orders = np.array([p.N for p in self.preds])
              for p in self.preds:
                  p.changeNmax(p.NMAX+1)
              # feed the new point at t_{n+1} to increase the stencil and increment the order
              self._advancePredictors(t=tnp1, coupling_vars=coupling_vars_np1)
              new_orders = self._getPredOrders()
              assert all((new_orders-old_orders)==1)
              bConverged2, niter2, ysol2 = self.perform_step(y0=yhist[i-1], t=tn,
                                                      dt=dt, bDebug=False,
                                                      bPrint=(nPrintLevel>2),
                                                      rtol=rtol/20,
                                                      atol_iter=rtol/5,
                                                      rtol_iter=rtol/5)
              if not bConverged2:
                  self.logger.log(INTEGRATION_DETAIL3,"\t embedded solution did not converge")
                  raise WRnonConvergence()
              else:
                  self.logger.log(INTEGRATION_DETAIL3,"\t embedded solution converged")
              coupling_vars_np1_2 = self._getCouplingVars(t=tnp1, y=ysol2)
              self._updatePredictors(t=tnp1, coupling_vars=coupling_vars_np1_2)
              embedded_preds = self._backupPredictors()
              self._restorePredictors(main_preds)
              
        except RuntimeError as e:
            self.logger.log(INTEGRATION_DETAIL2, ' issue during WR iteration --> lowering time step')
            self.logger.log(INTEGRATION_DETAIL3, f' error was {e}')
            step_info.append((tn,dt,self._getPredOrders(),OTHEREXCEPTION))
            dt=dt/4
            nsteps_failed+=1
            nsteps_total+=1
            consecutive_fails+=1
            continue
        except WRnonConvergence:
            self.logger.log(INTEGRATION_DETAIL3, 'WR Loop has not converged')
            if not (bConverged is None):
                if bConverged:
                    assert not bConverged2
                    step_info.append((tn,dt,self._getPredOrders(),WRNONCONVERGENCE2))
                else:
                    step_info.append((tn,dt,self._getPredOrders(),WRNONCONVERGENCE))
            else: # WRnonConvergence should only be raised from this function
                msg = 'Non convergence has occured but has not been handled properly'
                self.logger.critical(msg)
                raise Exception(msg)

            dt=dt/4
            nsteps_failed+=1
            nsteps_total+=1
            consecutive_fails+=1
            continue
        # TODO: order adaptation --> lower order ?

        # Update predictors with the new coupling variables
        # self._updatePredictors(t=tnp1, coupling_vars=coupling_vars_np1)
        # print('update')
        # print(' self.preds[0].x=', self.preds[0].x)
        # print(' self.preds[0].y=', self.preds[0].y)
        # self._restorePredictors(initial_extraps) # safety first
        # self._advancePredictors(t=tnp1, coupling_vars=coupling_vars_np1)
        # print('restore and advance')
        # print(' self.preds[0].x=', self.preds[0].x)
        # print(' self.preds[0].y=', self.preds[0].y)
        #### Error control
        # compute error based on the comparison of the extrapolated and
        # converged coupling variables
        dt_opts = []
        dt_opts2 = []
        self.logger.log(INTEGRATION_DETAIL3,"\t estimating coupling errors")
        for ii in range(len(self.preds)):
            if self.embedded_method:
              new_pred = embedded_preds[ii]
              old_pred = main_preds[ii]
            else:
              new_pred = main_preds[ii]
              old_pred = initial_extraps[ii]
  
            dt_opts.append( old_pred.eval_optimal_timestep_single(
                                        ref_value=new_pred.eval_single(tnp1),
                                        time=tnp1, dt=dt, order=None) )
            dt_opts2.append( new_pred.eval_optimal_timestep_single(
                                        ref_value=old_pred.eval_single(tnp1),
                                        time=tnp1, dt=dt, order=None) )
        
        dt_opts3 = np.maximum(dt_opts,dt_opts2)
        dt_opt = min(dt_opts3)
        self.logger.log(INTEGRATION_DETAIL3, f'\tdt_opt = {dt_opt:.3e} s')

        bAccepted = dt < TOLERANCE_FACTOR_DTOPT*dt_opt
        if bAccepted:
          step_info.append((tn,dt,self._getPredOrders(),ACCEPTED,dt_opts,dt_opt))
          self.logger.log(INTEGRATION_DETAIL, '\t==> accepted step')
          yhist.append( np.copy(ysol) )
          couplingvar_hist.append( np.hstack([p.eval_single(tnp1) for p in self.preds]) )
          thist.append( tnp1 )
          iter_hist.append(niter)
          if self.embedded_method:
              iter_hist2.append(niter2)
          tn=tnp1
          nsteps_accepted+=1
          self._restorePredictors(initial_extraps)
          self._advancePredictors(t=tnp1, coupling_vars=coupling_vars_np1)
          p_hist.append([p.N for p in self.preds])
          if adaptive_order:
            raise Exception('adaptive order currently seems to be broken')
            for p in self.preds:
              new_order= p.suggest_order_error_based(ref_value=old_pred.eval_single(tnp1), time=tnp1)
              p.setCurrentN(new_order)
        else:
          self.logger.log(INTEGRATION_DETAIL3,"\t step refused (error too large)")
          step_info.append((tn,dt,self._getPredOrders(),ERRORTOOHIGH,dt_opts,dt_opt))
          nsteps_rejected+=1
          ntry+=1

        nsteps_total+=1
        dt_opt = SAFETY_FACTOR*dt_opt
        dt_original = dt
        assert dt_opt>0
        dt=dt_opt
        if dt_opt<MINRELSTEP*dt_original:
          self.logger.log(INTEGRATION_DETAIL3, 'time step limited by maximum reduction factor')
          dt = MINRELSTEP*dt_original

        if dt_opt>MAXRELSTEP*dt_original:
          self.logger.log(INTEGRATION_DETAIL3, 'time step limited by maximum increase factor')
          dt = MAXRELSTEP*dt_original

        if bAccepted:
          dt_end = t_span[1]-tnp1
        else:
          dt_end = t_span[1]-tn
        if dt>dt_end:
          self.logger.log(INTEGRATION_DETAIL3, 'time step limited by final time')
          dt = dt_end

        if dt>max_step:
          self.logger.log(INTEGRATION_DETAIL3, 'time step limited by maximum allowed time step')
          dt = max_step

        if dt<0:
          msg = "time step has become negative"
          self.logger.critical(msg)
          raise Exception(msg)

    self.logger.log(INTEGRATION, 'Adaptive coupled integration has successfully reached end time')
    tend = pytime.time()

    out = OdeResult()
    out.t = np.array(thist)
    out.y = np.array(yhist).T
    assert out.y.ndim == 2
    out.z = np.array(couplingvar_hist).T
    out.success = True
    out.message = 'Success'
    out.WR_iters = np.array( iter_hist )
    if self.embedded_method:
        out.WR_iters2 = np.array( iter_hist2 )
    out.nsteps_total = nsteps_total
    out.nsteps_rejected = nsteps_rejected
    out.nsteps_accepted = nsteps_accepted
    out.nsteps_failed = nsteps_failed
    out.CPUtime = tend-tstart
    out.p_hist = np.array(p_hist)
    out.step_info = step_info

    return out


  def fixed_point(self, func, x0, args=(), xtol=1e-8, maxiter=500, method='del2'):
    """
    (specialised from Scipy's routine)
    Find a fixed point of the function.
    Given a function of one or more variables and a starting point, find a
    fixed point of the function: i.e., where ``func(x0) == x0``.
    Parameters
    ----------
    func : function
        Function to evaluate.
    x0 : array_like
        Fixed point of function.
    args : tuple, optional
        Extra arguments to `func`.
    xtol : float, optional
        Convergence tolerance, defaults to 1e-08.
    maxiter : int, optional
        Maximum number of iterations, defaults to 500.
    method : {"del2", "iteration"}, optional
        Method of finding the fixed-point, defaults to "del2",
        which uses Steffensen's Method with Aitken's ``Del^2``
        convergence acceleration [1]_. The "iteration" method simply iterates
        the function until convergence is detected, without attempting to
        accelerate the convergence.
    References
    ----------
    .. [1] Burden, Faires, "Numerical Analysis", 5th edition, pg. 80
    Examples
    --------
    >>> from scipy import optimize
    >>> def func(x, c1, c2):
    ...    return np.sqrt(c1/(x+c2))
    >>> c1 = np.array([10,12.])
    >>> c2 = np.array([3, 5.])
    >>> optimize.fixed_point(func, [1.2, 1.3], args=(c1,c2))
    array([ 1.4920333 ,  1.37228132])
    """
    from scipy._lib._util import _asarray_validated, _lazywhere
    def _del2(p0, p1, d):
        return p0 - np.square(p1 - p0) / d
    def _relerr(actual, desired):
        return (actual - desired) / desired

    use_accel = {'del2': True, 'iteration': False}[method]
    self.logger.log(FIXEDPOINT, f'fixed-point with method "{method}" and xtol={xtol:.2e}')
    x0 = _asarray_validated(x0, as_inexact=True)
    if x0.size>1 and use_accel:
        raise Exception("only scalar equations can be solved with Aitken's' acceleration")
    p0 = x0
    for i in range(maxiter):
        p1 = func(p0, *args)
        if use_accel:
            p2 = func(p1, *args)
            d = p2 - 2.0 * p1 + p0
            p = _lazywhere(d != 0, (p0, p1, d), f=_del2, fillvalue=p2)
        else:
            p = p1
        relerr = _lazywhere(p0 != 0, (p, p0), f=_relerr, fillvalue=p) / xtol
        # relerr = (p-p0)/(xtol + xtol*abs(p0))
        self.logger.log(FIXEDPOINT, f'fixed-point iteration {i}, rel err={relerr}')
        if np.all(np.abs(relerr) < 1):
            self.logger.log(FIXEDPOINT, f'fixed-point converged after {i} iterations')
            return p, i
        p0 = p
    msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
    self.logger.critical(msg)
    raise RuntimeError(msg)

"""
  def zzzzz_perform_step(self, y0, t, dt, atol_iter=None, rtol_iter=None, bDebug=False, bPrint=True, rtol=None):
        # Perform the WR iteration process for a single time step
        print('cosim: performing step')
        bConverged = False
        tn = t # current starting time
        tnp1 = t+dt # next coupling time

        ynp1_kp1 = y0 # TODO: better guess
        outs_kp1 = None # backup of the previous dense outputs

        last_iter_secant = 0
        if atol_iter is None:
          atol_iter = self.waveform_tolerance
        if rtol_iter is None:
          rtol_iter = self.waveform_tolerance

        pred_coupling_vars = np.array([p.eval_single(tnp1, allow_outside=True) for p in self.preds])[:,0]

        ## Since we are at the first iteration, we only have data up to time t
        # and extrapolate on [t, t+dt]
        # To do so:
        # - we first register the solution at time t once at the first iteration
        #   of the new time step (which starts from t and goes to t+dt)
        # - then we evaluate the extrapolation at t+dt and register a fake point
        #   at t+dt with these values (this drops the oldest interpolation point,
        #   but does not affect the prediction polynomial)
        # - then, for the other iterations of the current time step, we simply
        #   update these last data points with the current iteration's values,
        #   so that we work in interpolation mode instead !

        #TODO: for the embedded method, be more clever !
        # 1 - We append these points
        assert not (self.preds[0].x is None), 'predictors have not been initialised !!!'
        if abs((self.preds[0].x[0] - tnp1)/dt) < 1e-2:
            if not self.embedded_method:
                raise Exception('Time tnp1 seems to have already been registered in the predictors...')
            # else we do nothing here, since the embedded approach already provides the proper perdictor setup
        else:
            # bPredNmaxedOut = [p.N<p.NMAX for p in self.preds]
            oldNs = [p.N for p in self.preds]
            self._advancePredictors(t=tnp1, coupling_vars=pred_coupling_vars)
            # discard the oldest point so that the polynomial is unaffected (in particular its order...)
            # however, we must set N back to its correct value after the WR iterations are done and the step is accepted
            for oldN,pred in zip(oldNs,self.preds):
                pred.setCurrentN(oldN)
            #TODO: for the Milne device, be more clever !
            # for (bNeedsCorrection, pred) in zip(bPredNmaxedOut, self.preds):
            #     if bNeedsCorrection:
            #         pred.decreaseN() # to avoid using the oldest point, which is now "redundant"

        ### WR iterations to converge the present coupling step
        iterated_values = np.zeros((self.NITER_MAX+1,len(self.preds)))
        iterated_values[0,:] = pred_coupling_vars
        nerrors=2
        errors = np.zeros((self.NITER_MAX,nerrors))
        coupling_vars_kp1 = np.array([p.eval_single(tnp1)[0] for p in self.preds])

        # TODO: automatic error control for all predicted variables + user-defined variables (which are not predicted)
        # TODO: complex step once to identify the dependence (boolean) of the errors on the predictors, so that the error orders are roughly known ?
        for i_iter in range(1,self.NITER_MAX+1):
          if bPrint:
            print('    iter {}'.format(i_iter))
          ynp1_k=ynp1_kp1
          outs_kp=outs_kp1

          # if i_iter>1:
          #     # Update predictors
          #     self._updatePredictors(t=tnp1, coupling_vars=self._getCouplingVars(t=tnp1, y=ynp1_k))
          #     # TODO: ça semble inutile, puisque c'est fait dans _step_forward
          # Compute the new sampling points for the predicted variables
          if self.interfaceSolver == 'secant':
              # raise Exception('does not work well ?')
              if (i_iter<4) or (i_iter-last_iter_secant<4): # fixed-point
                  coupling_vars_k = self._getCouplingVars(t=tnp1, y=ynp1_k)
              else: # secant update on 0 = F(x) for the fixed-point problem x=g(x) <=> F(x)=x-g(x)
                xk   = iterated_values[i_iter-2,:]
                xkm1 = iterated_values[i_iter-3,:]
                xkp1 = iterated_values[i_iter-1,:]

                g_xkm1 = xk
                g_xk   = xkp1

                f_xk = xk-g_xk
                f_xkm1 = xkm1-g_xkm1

                fprime = (xk-xkm1)/(f_xk-f_xkm1)
                secant_values_predict = xk - f_xk/fprime
                if bPrint:
                  print('fprime=',fprime)
                  print('secant_values_predict=',secant_values_predict)
                coupling_vars_k = (secant_values_predict[jj] for jj in range(len(xk)))
                last_iter_secant = i_iter

          elif self.interfaceSolver == 'newton':
              if self.coupler.bUseDenseOutput:
                raise Exception('Newton method cannot be used in conjunction with dense output iterations')

              rtol_jac = 1e-3
              atol_jac = rtol_jac

              if i_iter>1:
                  coupling_vars_k = self._getCouplingVars(t=tnp1, y=ynp1_k)
              else:
                  coupling_vars_k = np.array([p.eval_single(tnp1)[0] for p in self.preds])

              h_pert = []
              for s in coupling_vars_k:
                  if abs(s)>2*atol_jac:
                      h_pert.append( rtol_jac * s )
                  else:
                      if s>=0:
                          h_pert.append(atol_jac)
                      else:
                          h_pert.append(-atol_jac)

              # backup predictors
              old_preds = self._backupPredictors() # to keep the original extrapolated values
              # TODO: improve by just keeping the values for each possible order

              # get unperturbed result
              ynp1_kp1_unpert, outs_kp1_unpert = self._step_forward(yn=y0,
                                                                t=tn,
                                                                dt=dt,
                                                                values_last_iterate=coupling_vars_k,
                                                                last_outs=outs_kp, rtol=rtol)
              coupling_vars_unpert = self._getCouplingVars(t=tnp1, y=ynp1_kp1_unpert)
              resid_unpert = coupling_vars_unpert - coupling_vars_k


              def G(x):
                # fonction dont on recherche le point fixe
                self._restorePredictors(old_preds)
                y, outs = self._step_forward(yn=y0, t=tn, dt=dt, values_last_iterate=x, last_outs=None, rtol=rtol)
                coupling_vars = self._getCouplingVars(t=tnp1, y=y)
                return coupling_vars

              def getResid(x):
                # pour trouver le point fixe de G par une méthode de Newton
                print('getting residuals at x[0]=',x[0])
                g = G(x)
                return g - x

              # temp = scipy.optimize.newton(func=getResid, x0=coupling_vars_unpert, fprime=None, args=(),
              #                            tol=1e-08, maxiter=50, fprime2=None, x1=None, rtol=1e-8,
              #                            full_output=False, disp=True)
              # print(temp)

              # compute Jacobian by finite differences
              jacobian = np.zeros((coupling_vars_k.size, coupling_vars_k.size))
              dukp1_duk = np.zeros((coupling_vars_k.size, coupling_vars_k.size))
              for ipert in range(coupling_vars_k.size):
                  # restore old predictors
                  self._restorePredictors(old_preds)

                  values_last_iterate_pert = np.copy(coupling_vars_k)
                  values_last_iterate_pert[ipert] = values_last_iterate_pert[ipert] + h_pert[ipert]

                  ynp1_kp1_pert, outs_kp1_pert = self._step_forward(yn=y0,
                                                                    t=tn,
                                                                    dt=dt,
                                                                    values_last_iterate=values_last_iterate_pert,
                                                                    last_outs=outs_kp, rtol=rtol)

                  coupling_vars_pert = self._getCouplingVars(t=tnp1, y=ynp1_kp1_pert)
                  # jacobian[:,ipert] = (coupling_vars_pert - coupling_vars_unpert) / h_pert[ipert]
                  resid = coupling_vars_pert - values_last_iterate_pert

                  if 1: # first-order
                      jacobian[:,ipert] = (resid- resid_unpert) / h_pert[ipert]
                      dukp1_duk[:,ipert] = (coupling_vars_pert- coupling_vars_unpert) / h_pert[ipert]
                  else: # 2nd-order centered
                      self._restorePredictors(old_preds)
                      values_last_iterate_pert = np.copy(coupling_vars_k)
                      values_last_iterate_pert[ipert] = values_last_iterate_pert[ipert] - h_pert[ipert]
                      ynp1_kp1_pert, outs_kp1_pert = self._step_forward(yn=y0,
                                                                        t=tn,
                                                                        dt=dt,
                                                                        values_last_iterate=values_last_iterate_pert,
                                                                        last_outs=outs_kp,
                                                                        rtol=rtol)

                      coupling_vars_pert2 = self._getCouplingVars(t=tnp1, y=ynp1_kp1_pert)
                      resid2 = coupling_vars_pert2 - values_last_iterate_pert
                      jacobian[:,ipert] = 0.5 * (resid - resid2) / h_pert[ipert]
                      dukp1_duk[:,ipert] = 0.5*(coupling_vars_pert- coupling_vars_pert2) / h_pert[ipert]



              # jacobian = np.eye((coupling_vars_k.size)) + jacobian
              print(' Newton')
              print('    h_pert:\n', h_pert)
              print('    Jacobian:\n',  jacobian)
              print('    dukp1_duk:\n', dukp1_duk)
              # Newton
              #  résidual of the fixed point: coupling_vars = f(coupling_vars)
              print('    residuals:\n', resid_unpert)
              # increment = -scipy.linalg.solve(a=jacobian, b=resresid_unpert
              increment = -scipy.linalg.pinv(jacobian).dot( resid_unpert )
              print('    increment:\n', increment)

              if 1: #alternate formulation : ukp1 = g(uk)
                  print('alternative newton:')
                  # fixed-point attained with linearisation (newton)
                  jacobian = np.eye(jacobian.shape[0]) - dukp1_duk
                  resid = coupling_vars_k - coupling_vars_unpert
                  print('    Jacobian:\n', jacobian)
                  # Newton
                  #  résidual of the fixed point: coupling_vars = f(coupling_vars)
                  print('    residuals:\n', resid_unpert)
                  # increment = -scipy.linalg.solve(a=jacobian, b=resresid_unpert
                  increment = -scipy.linalg.pinv(jacobian).dot( resid_unpert )
                  print('    increment:\n', increment)


              coupling_vars_k = coupling_vars_k + increment
              print('    coupling_vars:\n', coupling_vars_k)

              # restore old predictors
              self._restorePredictors(old_preds)

          elif self.interfaceSolver == 'fixed-point': # basic fixed-point iterations
              coupling_vars_k = coupling_vars_kp1 #self._getCouplingVars(t=tnp1, y=ynp1_k)
          else:
              raise Exception('Interface solver {} unknown'.format(self.interfaceSolver))

          # integrate subsytems
          ynp1_kp1, outs_kp1 = self._step_forward(yn=y0, t=tn, dt=dt,
                                           values_last_iterate=coupling_vars_k,
                                           last_outs=outs_kp, rtol=rtol)
          if np.any(np.isnan(ynp1_kp1)):
            raise Exception('NaNs have appeared in the global state vector')

          # keep track of the evolution of the predicted values (for debugging / analysis)
          # for ii in range(len(self.preds)):
          #   iterated_values[i_iter,ii] = self.preds[ii].eval_single(tnp1)
          coupling_vars_kp1 = self._getCouplingVars(t=tnp1, y=ynp1_kp1)
          iterated_values[i_iter,:] = coupling_vars_kp1

          if bDebug:
            plt.figure()
            pred = self.preds[0]
            plt.plot(pred.getX(), pred.getY()[0,:], label='data', marker='o', color='r')
            xtest = np.linspace(tn-4*dt,tn+2*dt,100)
            plt.plot(xtest, pred.eval_vector(xtest, allow_outside=True)[0,:])
            plt.grid()
            plt.ylabel('prediction (coupling var 0)')
            plt.xlabel('t')
            plt.title('At iteration {}'.format(i_iter))
            plt.show()

          ### Monitor convergence ###
          # error on the solution vectors
          errors[i_iter-1,:], error_norm_for_cv = self._getErrors(y=ynp1_kp1, yref=ynp1_k,
                                                                  t=tnp1, i_iter=i_iter,
                                                                  atol=atol_iter, rtol=rtol_iter,
                                                                  bPrint=bPrint)

          # monitor convergence rate
          if ((i_iter>=2) and (self.checkConvergenceRate)):
            # estimate convergence rate
            conv_rate = errors[i_iter-2,0]/errors[i_iter-1,0]
            nrequired_iter = -np.log(1 / errors[i_iter-1,0]) / np.log(conv_rate)
            nrequired_iter = nrequired_iter + i_iter
            if bPrint:
              print('       ****conv_rate=',conv_rate)
              print('       ******* required_iter=', nrequired_iter)
            if (conv_rate<1) or (nrequired_iter>self.NITER_MAX):
              if bDebug:
                  plt.figure()
                  plt.semilogy(range(1,i_iter+1), errors[:i_iter,0], label='Ts', marker='+')
                  plt.semilogy(range(1,i_iter+1), errors[:i_iter,1], label='Tfield', marker='+')
                  plt.grid()
                  plt.legend()
                  plt.title('Evolution of relative errors')
                  plt.xlabel('iter')
                  plt.ylabel('$\epsilon$')
                  plt.grid()
                  plt.show()

                  for j in range(iterated_values.shape[1]):
                    plt.figure()
                    plt.plot( iterated_values[:i_iter,j], iterated_values[1:i_iter+1,j], marker='+', color='tab:blue', label='iterates')
                    plt.plot( iterated_values[0,j],         iterated_values[1,j],        marker='*', color='tab:green', label='initial value')
                    plt.plot( iterated_values[i_iter-1,j],  iterated_values[i_iter,j], marker='*', color='tab:red', label='final value')
                    # the line "xp1=xp" is where the solution to the fixed point problem lies !
                    plt.plot( [iterated_values[0,j], iterated_values[i_iter,j]],
                              [iterated_values[0,j], iterated_values[i_iter,j]],
                              marker=None, color='tab:orange', linestyle='--', label='1:1')
                    # see where it interesects with a fitted affine function
                    poly = np.polyfit(iterated_values[:i_iter,j], iterated_values[1:i_iter+1,j], deg=1)
                    plt.plot(                 iterated_values[:i_iter,j],
                             np.polyval(poly, iterated_values[:i_iter,j]), marker='+', color=[0,0,0,0.5],
                             label='affine fit', linestyle='--')
                    print('var {}, poly={}'.format(j,poly))

                    # Predicted solutions following a pointwise affine model
                    if 0:
                        for i in range(0, i_iter-1):
                            # assume xnp1 = g(xp) = a*xp + b
                            a = (iterated_values[i+2,j] - iterated_values[i+1,j]) / (iterated_values[i+1,j] - iterated_values[i,j])
                            b = iterated_values[i+1,j] - a * iterated_values[i,j]
                            xsol = b/(1-a)
                            plt.plot(xsol,xsol, color=[0,0,0], marker='o', label=None)
                    plt.legend()
                    plt.title('Fixed-Point problem for predictor {}'.format(j))
                    plt.xlabel(r'$x_n$')
                    plt.ylabel(r'$x_{n+1} = g(x_n)$')
                    plt.grid()

                  # compute jacobian of residuals, a posteriori
                  # resid = np.diff(iterated_values[0,:])
                  for i in range(0, i_iter):
                      resid = iterated_values[i+1,:] - iterated_values[i,:]
                      dvar = iterated_values[i+2,:] - iterated_values[i-1,:]
                      jac = resid/dvar # ??? (suppose diagonal jacobian)
                      print('jac=', jac)
                  raise Exception('debug')
              if (not self.interfaceSolver=='secant'):
                print('WR will not converge')
                return False, None, None
                # raise WRnonConvergence('WR will not converge')

          # convergence criterion
          if error_norm_for_cv < 1.:
            if bPrint:
              print(f'         ==>  CONVERGED after {i_iter} iterations !')
            bConverged = True
            break

          # raise Exception('stop debug')

        print('cosim: step performed')
        return bConverged, i_iter, ynp1_kp1

"""

def process_step_info(outCosim):
    """ Post-process step rejections and other information """
    error_codes = []
    step_info = {}
    temp=(
            ('WRNONCONVERGENCE', 1),
            ('WRNONCONVERGENCE2',2),
            ('OTHEREXCEPTION',   5),
            ('ERRORTOOHIGH',     6),
            ('ACCEPTED',         0),
         )
    error_codes = [a[1] for a in temp]

    for failmode, value in  temp:
        step_info[failmode] = {"error_code": value,
                               "tn": [],
                               "dt": [],
                               "orders": [],
                               "dt_opts": [],
                               "dt_opt": [],
                               }

    for d in outCosim.step_info:
        error_code = d[3]
        ierr = error_codes.index(error_code)
        code = temp[ierr][0]
        tn = d[0]
        dt = d[1]
        orders = d[2]
        if len(d)>4:
            dt_opts = d[4]
            dt_opt  = d[5]

        assert step_info[code]['error_code']==error_code
        step_info[code]["tn"].append(tn)
        step_info[code]["dt"].append(dt)
        step_info[code]["orders"].append(orders)

        if len(d)>4:
            step_info[code]["dt_opts"].append(dt_opts)
            step_info[code]["dt_opt"].append(dt_opt)

    for key1 in step_info.keys():
        for key2 in step_info[key1].keys():
            if isinstance(step_info[key1][key2], list):
                step_info[key1][key2] = np.array(step_info[key1][key2])
    return step_info
