#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:54:51 2023

@author: lfrancoi
"""




import numpy as np
from src.lib.rhapsopy.coupling import FIXEDPOINT, WRnonConvergence, ExceptionWhichMayDisappearWhenLoweringDeltaT
from src.lib.rhapsopy.accelerators.base import BaseFixedPointSolver
import scipy.optimize
from scipy.optimize.nonlin import NoConvergence

class JFNKSolver(BaseFixedPointSolver):
  def __init__(self, logger):
    super().__init__(logger)

  def solve(self, fun, x0, ftol, rtol, maxiter, args=()):
    global ncalls
    ncalls = 0
    def resfun(x): # residuals
      global ncalls
      ncalls += 1
      # print('\t call {}, x='.format(ncalls), x)
      if ncalls>maxiter:
        raise NoConvergence()
      return x-fun(x)

    try:
      if 1: # no preconditionner
        inner_M = None

      else: # jacobian-based preconditionner
        # Jac = scipy.optimize._numdiff.approx_derivative(fun=resfun, x0=x0,
        #                      method='2-point', rel_step=1e-6, bounds=(0.,np.inf)
        #                      )
        # inner_M = np.linalg.inv(Jac)

        from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
        from scipy.optimize.nonlin import InverseJacobian
        jac = BroydenFirst()
        inner_M = KrylovJacobian(inner_M=InverseJacobian(jac))

      sol = scipy.optimize.newton_krylov(F=resfun, xin=x0, iter=None, rdiff=max(1e-6,rtol/10),
                               method='lgmres', inner_maxiter=maxiter+1, inner_M=inner_M, outer_k=100,
                               verbose=True, maxiter=maxiter, f_tol=1e-30,
                               f_rtol=None, x_tol=rtol, x_rtol=rtol, tol_norm=None,
                               line_search=None,
                               callback=None)
    except NoConvergence as e:
      print(e)
      msg = "JFNK Failed to converge after {} calls".format(ncalls)
      self.logger.critical(msg)
      raise WRnonConvergence(msg)
      
    except (OverflowError, ValueError) as e:
      print(e)
      msg = "JFNK internal error after {} calls ({})".format(ncalls,e)
      self.logger.critical(msg)
      raise ExceptionWhichMayDisappearWhenLoweringDeltaT(msg)
      
    return sol, ncalls