#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:38:59 2020

Python implementation of an exact damped Newton scheme, with LU factorsation of the Jacobian

@author: Laurent
"""
import scipy.linalg
import scipy.optimize
import numpy as np
from src.lib.rhapsopy.accelerators import NewtonSolver
from src.lib.rhapsopy.coupling import FIXEDPOINT, WRnonConvergence

NORM_ORD = None
def computeStepNorm(dx,x,rtol,atol):
    """ Step norm is < 1 at convergence"""
    return np.linalg.norm(dx/(atol+rtol*abs(x)), ord=NORM_ORD) / np.sqrt(dx.size)

def computeResidualNorm(res):
    return np.linalg.norm(res, ord=NORM_ORD) / np.sqrt(res.size)

class DampedNewtonSolver(NewtonSolver):
  def __init__(self, logger):
      super().__init__(logger)
      self.jac = None
      self.NJAC_MAX = 10
      self.TAU_MIN = 1e-4
      self.TAU_SAFE = 0.1 # If damping falls below that but step is accepted, Jacobian will be updated
      self.GOOD_CONVERGENCE_RATIO_STEP = 0.5
      self.GOOD_CONVERGENCE_RATIO_RES  = 0.5
      self.NMAX_BAD_STEPS = 1

  def _solve_lin(self,A,b):
    return np.linalg.solve(A,b)
    #TODO: optimise with LU-decomposition ?

  def solve(self, fun, x0, ftol, rtol, maxiter, **kwargs):
    resfun = lambda x: x-fun(x) # residuals
    return self._solve(resfun, x0, ftol, rtol, maxiter, **kwargs)

  def _solve(self, fun, x0, ftol, rtol, maxiter, **kwargs):

    njev=0
    nfev=0
    niter=0
    atol=rtol
    nlinsolve = 0

    # initiate computation
    tau=1.0 # initial damping factor #TODO: warm start information ?
    x=np.copy(x0)

    bSuccess=False
    bUpdateJacobian = False
    bJacobianAlreadyUpdated = False # True if the last jacobian was compouted for the current value of x
                                   # (ie if we ask to recompute it for the same x, there is a problem)
    nforced_steps = 0
    res = None

    old_dx_norm = None
    old_res_norm = None
    dx = None
    bStepComputedAtCurrentX = False
    self.logger.log(FIXEDPOINT, f'Starting damped Newton loop with ftol={ftol:.1e}, rtol={rtol:.1e}')
    while True: # cycle until convergence
        bAcceptedStep=False
        if bUpdateJacobian or (self.jac is None):
            if bJacobianAlreadyUpdated:
              if nforced_steps >= self.NMAX_BAD_STEPS:
                  msg = '\t/!\ the jacobian has already been computed for this value of x --> convergence seems impossible, even with forced bad steps...'
                  break

              # msg = '\t forcing bad steps has not yet been implemented'
              # break
              self.logger.log(FIXEDPOINT, '\t    --> forcing one step and retrying')
              # raise NotImplementedError()
              tau = 1.
              x = x - tau*dx
              res = fun(x)
              nfev+=1
              res_norm = computeResidualNorm(res)
              bStepComputedAtCurrentX = False
              bJacobianAlreadyUpdated = False
              nforced_steps += 1

            elif njev > self.NJAC_MAX:
              msg = '\ttoo many Jacobian evaluations'
              break

            else: # update jacobian and dx
              self.logger.log(FIXEDPOINT, '\tupdating Jacobian...')
              res, self.jac = self._computeJacobian(fun, x, h=max(1e-8, min(1e-2, rtol/5)), bReturnFval=True)
              nfev+=1
              njev += 1
              self.logger.log(FIXEDPOINT, '\tJacobian updated')
              
              if 0: # plot sparsity pattern in color
                  # import pdb; pdb.set_trace()
              #%%
                  valueDres = abs(self.jac)
                  valueDres[valueDres==0] = np.nan
                  
                  # each line is divided by its max component
                  valueDres_rel = valueDres.copy()
                  for iline in range(valueDres_rel.shape[0]): # remove identity part to isolate off-diagonal blocks
                      valueDres_rel[iline,iline] = np.nan
                  for iline in range(valueDres_rel.shape[0]):
                    maxval = np.nanmax(valueDres_rel[iline,:])
                    if maxval !=0:
                        valueDres_rel[iline,:] = valueDres_rel[iline,:] / maxval

                  valueDres_rel = np.log10(valueDres_rel)
                  valueDres = np.log10(valueDres)

            
                  lowerLim = np.nanmin(valueDres[np.isinf(valueDres)==0])
                  upperLim = np.nanmax(valueDres[np.isinf(valueDres)==0])
                  valueDres[np.isinf(valueDres)] = np.nan
            
                  import matplotlib.pyplot as plt
                  fig = plt.figure()
                  ax = fig.add_subplot(111)
                  plt.imshow(valueDres,interpolation='none', cmap=plt.cm.get_cmap('jet', 20)) #plt.cm.ocean)
                  plt.colorbar(extend='both')
                  plt.clim(lowerLim, upperLim)
                  
                  plt.contour(1.*(valueDres_rel>-3), levels=[0.5], linestyles=':', colors=([1,1,1],))
                  
                  ax.axis('scaled')
                  ax.set_title('Jacobian values (negative log scale)')
            
                  # ax.set_xticks(major_ticks_x)
                  # ax.set_xticks(minor_ticks, minor=True)
                  # ax.set_yticks(major_ticks_y)
                  # ax.set_yticks(minor_ticks, minor=True)
                  ax.grid(which='both')
                  ax.grid(which='minor', alpha=0.5)
                  ax.grid(which='major', alpha=0.9)
            
                  plt.show()
              #%%
              res_norm = computeResidualNorm(res)
              bUpdateJacobian = False
              bJacobianAlreadyUpdated=True
              # tau = 1. # TODO: bof ?

        niter+=1
        if niter > maxiter:
          msg='\ttoo many iterations'
          break

        if res is None: # initial residuals
          res = fun(x)
          nfev+=1
          res_norm = computeResidualNorm(res)

        if res_norm < ftol:
            self.logger.log(FIXEDPOINT, 'residual norm sufficiently small')
            bSuccess = True
            break

        # Compute Newton step
        if not bStepComputedAtCurrentX:
          dx = self._solve_lin(self.jac, res)
          nlinsolve+=1
          dx_norm = computeStepNorm(dx,x,rtol,atol)
        self.logger.log(FIXEDPOINT, f'it={niter}, ||res||={res_norm:.1e}, ||dx||={dx_norm:.1e}, tau={tau:.1e}')

        # print('x=',x)
        # print('dx=',dx)
        # print('res=',res)
        # import sys
        # sys.stdout.flush()

        if dx_norm < 1:
            self.logger.log(FIXEDPOINT, f'step norm sufficiently small')
            bSuccess = True
            break


        # Compute new iterate (with damping)
        new_x = x - tau*dx

        new_res = fun(new_x)
        nfev+=1
        new_res_norm = computeResidualNorm(new_res)

        new_dx  = self._solve_lin(self.jac, new_res)
        nlinsolve+=1
        new_dx_norm = computeStepNorm(new_dx, new_x, rtol, atol)

        self.logger.log(FIXEDPOINT, '\t ||new dx||={:.1e}, ||new res||={:.1e}'.format(new_dx_norm, new_res_norm))
        if new_dx_norm < dx_norm:
            self.logger.log(FIXEDPOINT, '\tstep decrease is satisfying ({:.1e}-->{:.1e} with damping={:.1e})'.format(dx_norm, new_dx_norm, tau))
            bAcceptedStep=True
        elif new_res_norm < res_norm:
            self.logger.log(FIXEDPOINT, '\tresidual decrease is satisfying ({:.1e}-->{:.1e} with damping={:.1e})'.format(res_norm, new_res_norm, tau))
            bAcceptedStep=True


        if bAcceptedStep: # current damping leads to a better iterate
            self.logger.log(FIXEDPOINT, '\tdamped step accepted (tau={:.1e})'.format(tau))

            # Jacobian update policy : is convergence fast enough ?
            if tau<self.TAU_SAFE:
              self.logger.log(FIXEDPOINT, '\t successful damping is considered too small --> jacobian will be updated')
              bUpdateJacobian = True
            else:
              if new_dx_norm/dx_norm > self.GOOD_CONVERGENCE_RATIO_STEP:
                  self.logger.log(FIXEDPOINT, '\tslow ||dx|| convergence (ratio is {:.2e})) --> asking for jac update'.format(new_dx_norm/dx_norm))
                  bUpdateJacobian = True
              elif new_res_norm/res_norm > self.GOOD_CONVERGENCE_RATIO_RES:
                  self.logger.log(FIXEDPOINT, '\tslow ||res|| convergence (ratio is {:.2e})) --> asking for jac update'.format(new_res_norm/res_norm))
                  bUpdateJacobian = True

            # Accept step
            x = new_x
            res = new_res
            res_norm = new_res_norm
            dx = new_dx
            dx_norm = new_dx_norm

            bStepComputedAtCurrentX = True
            bJacobianAlreadyUpdated=False # we have not yet computed the Jacobian for the new value of x

            # TODO: Rank-1 update of the Jacobian ?
            if 0:# tau==1. and not bUpdateJacobian: # no damping was applied, the Jacobian can be correctly updated
              # adapted from equation (1.60) of "A family of Newton Codes for Systems of Highly Nonlinear Equations" by Nowak and Deuflhard
              # --> seems to be useless in my test cases
              if dx_norm>1e-15:
                self.logger.log(FIXEDPOINT, '\t rank-1 update of the Jacobian matrix')
                self.jac = self.jac + np.outer(res,dx)/dx_norm

            # TODO: increase tau if convergence rate was good ?
            # tau = min((tau*5., 1.))
            tau = min((tau**0.5, 1.))

        else:
            # the step is rejected, we must lower the damping or update the Jacobian
            # TODO: better damping strategy
            tau = min(tau*tau, 0.5*tau)
            self.logger.log(FIXEDPOINT, '\tstep rejected: reducing damping to {:.3e}'.format(tau))
            if tau < self.TAU_MIN: # damping is too small, indicating convergence issues
                self.logger.log(FIXEDPOINT, '\tdamping is too low')
                bUpdateJacobian = True

    if bSuccess:
      self.logger.log(FIXEDPOINT, f'Newton converged (nfev={nfev}, njev={njev}, nlinsolve={nlinsolve})')
    else:
      self.logger.log(FIXEDPOINT, 'Newton failed:')
      self.logger.log(FIXEDPOINT, msg)
      raise WRnonConvergence(msg)
      # raise Exception('Newton did not converge')

    return x, niter

if __name__=='__main__':

    import logging
    logger = logging.getLogger("rhapsopy.test")
    logger.handlers = [] # drop previous handlers
    logger.setLevel(1) # no logging by default

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(1) # no filtering of logs
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    solver = DampedNewtonSolver(logger)

    import matplotlib.pyplot as plt
    print('=== testing the damped Newton solver ===\n')
    nprob=1 # choice of the test problem

    if nprob==0: # non-linear problem, with potential divergence if undamped
      def funplot(x):
          return np.array((np.arctan(x[0,:]), x[1,:]**3 + 0.6*x[1,:]))
      fun = lambda x: np.array((np.arctan(x[0]), x[1]**3 + 0.6*x[1]))
      # jacfun = lambda x: np.array( [[1/(1+x[0]**2), 0.], [0., 0.3+2*x[1]]] )
      x0 = np.array((0.5,0.5))

      # visualize the two residual function components
      plt.figure()
      xtest = np.linspace(-1,1,1000)
      xtest = np.vstack((xtest, xtest))
      plt.plot(xtest[0,:], funplot(xtest)[0,:], color='tab:orange', label='f1')
      plt.axvline(x0[0], color='tab:orange', label=None)
      plt.plot(xtest[0,:], funplot(xtest)[1,:], color='tab:blue', label='f2')
      plt.axvline(x0[1], color='tab:blue', label=None)
      plt.axhline(0., color=[0,0,0], label=None)

      plt.scatter(x0, fun(x0), marker='x', color='r', label=None)
      plt.legend()
      plt.xlabel('x')
      plt.ylabel('f(x)')
      plt.grid()

    elif nprob==1: ## quadratic problem - 1 var
      fun = lambda x: 2*x**2 + 0.5*x + 0.1 - 1
      # jacfun = lambda x: np.array([0.1*2*x + 0.5,])
      x0 = np.array((0.5,))

      plt.figure()
      xtest = np.linspace(-2,2,1000)
      plt.plot(xtest, fun(xtest))
      plt.scatter(x0, fun(x0), marker='x', color='r')
      plt.xlabel('x')
      plt.ylabel('f(x)')
      plt.grid()

    elif nprob==2: # quadratic problem - 2 vars (ellipse)
      aa = 100. #4e3
      bb = 1. #1e3
      fun =    lambda x: np.array( (aa*x[0]**2, x[1]**2))+ bb*x
      # jacfun = lambda x: np.array( [ [aa*2*x[0], 0.], [0., 2*x[1]] ] ) + bb*np.eye(x.size)
      x0 = np.array((0.5,0.5))
      # this problem hightlights the affine invariance property of the Newton method:
      # deforming the ellipsis by increasing "a" is not affecting the Newton path and convergence

    elif nprob==3: # linear problem
      fun = lambda x: x
      # jacfun = lambda x: np.eye(x.size)
      x0 = np.array((0.5,0.5))

    else:
      raise NotImplementedError()

    root, n_iter = solver._solve(fun=fun, x0=x0, ftol=1e-30, rtol=1e-8, maxiter=50)
