#!/usr/bin/env python3
# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:17:45 2023

@author: lfrancoi
"""
import numpy as np
from src.lib.rhapsopy.coupling import FIXEDPOINT, WRnonConvergence
from src.lib.rhapsopy.accelerators.base import BaseFixedPointSolver
import scipy

class NewtonSolver(BaseFixedPointSolver):
  def __init__(self, logger):
    super().__init__(logger)

  def _computeJacobian(self, fun, x, funx=None, h=1e-6, bReturnFval=False):
      nx = x.size
      Jac = np.empty((nx,nx))
      if funx is None:
          funx = fun(x)

      for i in range(nx):
          pert = x.copy()
          dh = (1 + abs(x[i])) * h
          if x[i]<0:
            dh= -dh
          pert[i] += dh

          fun_pert = fun(pert)
          Jac[:,i] = ( fun_pert - funx ) / dh
      if bReturnFval:
          return funx, Jac
      else:
          return Jac

  def solve(self, fun, x0, ftol, rtol, maxiter):
    resfun = lambda x: x-fun(x) # residuals
    newx  = x0.copy()
    it = 0
    bConverged = False
    self.logger.log(FIXEDPOINT, f'Starting Newton loop with ftol={ftol:.1e}, rtol={rtol:.1e}')
    # TODO: ftol is unused
    while not bConverged:
      it+=1
      if it>maxiter:
          self.logger.log(FIXEDPOINT, 'Maximum number of iterations reached')
          break

      x = newx.copy()
      res, Jac = self._computeJacobian(resfun, x, h=max(1e-8, rtol/5), bReturnFval=True)    

      # print('Jac=',Jac)

      dx = np.linalg.solve(Jac, -res)

      res_norm = self.compute_error_norm( res )
      dx_norm  = self.compute_error_norm( dx )
      newx = x + dx
      error = self.compute_error(x, newx, rtol=rtol)

      self.logger.log(FIXEDPOINT, f' it={it}, x={x}, dx={dx}, res={res}, jac={Jac}')
      self.logger.log(FIXEDPOINT, f' it={it}, ||res||={res_norm:.1e}, ||dx||={dx_norm:.1e}, ||err||={error:.1e}')
      # print('x=',x)
      # print('dx=',dx)
      # print('res=',res)
      # import sys
      # sys.stdout.flush()
      if error < 1.:
        bConverged = True
        break


    if not bConverged: # debug plots
        self.logger.log(FIXEDPOINT, 'Newton did not converge')
        if x0.size==1:
            import matplotlib.pyplot as plt
            var_test = np.linspace(0.2, 0.5, 20)
            fun_val = np.array([ fun(coupling_vars_k=np.array([v])) for v in var_test ])
            # import pdb; pdb.set_trace()
            plt.figure()
            plt.plot(var_test, var_test, linestyle='--', color=[0,0,0], label='y=x')
            plt.plot(var_test, fun_val, linestyle=':', color='tab:red', label='f(x)')
            plt.grid()
            plt.xlabel('Ts')
            plt.legend()

            getJac = lambda x, h:  self.computeJacobian(fun, x, h)
            h_test = np.logspace(-8,-1,10)
            jac_val = np.array([ getJac(x=x0, h=h) for h in h_test ])
            # import pdb; pdb.set_trace()
            plt.figure()
            plt.semilogx(h_test, jac_val[:,0,0], linestyle=':', color='tab:red', label='df/dx')
            plt.grid()
            plt.xlabel('Ts')
            plt.legend()
            plt.title('Jacobian')
            plt.ylim(-1,1)
            plt.show()
        # raise RuntimeError('Newton did not converge')
        raise WRnonConvergence('Newton did not converge')
    else:
        self.logger.log(FIXEDPOINT, ' --> Newton converged')
        return newx, it