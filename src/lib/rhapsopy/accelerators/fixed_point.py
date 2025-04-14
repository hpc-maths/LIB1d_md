#!/usr/bin/env python3
# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:20:26 2023

@author: lfrancoi
"""
import numpy as np
from src.lib.rhapsopy.coupling import FIXEDPOINT, WRnonConvergence
from src.lib.rhapsopy.accelerators.base import BaseFixedPointSolver
def array2exp(array, sformat='{:.2e}'):
  return '['+', '.join([sformat.format(a) for a in array])+']'

class FixedPointSolver(BaseFixedPointSolver):
  def __init__(self, logger):
    super().__init__(logger)
    self.method='iteration'

  def solve(self, fun, x0, ftol, rtol, maxiter, args=()):
    """
    (specialised from Scipy's routine)
    Find a fixed point of the function.
    Given a function of one or more variables and a starting point, find a
    fixed point of the function: i.e., where ``fun(x0) == x0``.
    Parameters
    ----------
    fun : function
        Function to evaluate.
    x0 : array_like
        Fixed point of function.
    args : tuple, optional
        Extra arguments to `fun`.
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
    """
    from scipy._lib._util import _asarray_validated, _lazywhere
    def _del2(p0, p1, d):
        return p0 - np.square(p1 - p0) / d

    use_accel = {'del2': True, 'iteration': False}[self.method]
    self.logger.log(FIXEDPOINT, f'fixed-point with method "{self.method}" and rtol={rtol:.2e}')
    x0 = _asarray_validated(x0, as_inexact=True)
    if x0.size>1 and use_accel:
        raise Exception("only scalar equations can be solved with Aitken's' acceleration")
    p0 = x0
    bPredictive = False
    if bPredictive:
      old_errs = np.ones((5,))*np.nan
    for i in range(maxiter):
        p1 = fun(p0, *args)
        if use_accel:
            p2 = fun(p1, *args)
            d = p2 - 2.0 * p1 + p0
            p = _lazywhere(d != 0, (p0, p1, d), f=_del2, fillvalue=p2)
        else:
            p = p1
        relerr = self.compute_error(x1=p, x2=p0, rtol=rtol)

        self.logger.log(FIXEDPOINT, f"fixed-point iteration {i}, rel err={relerr}")
        if np.abs(relerr) < 1:
            self.logger.log(FIXEDPOINT, f'fixed-point converged after {i} iterations')
            return p, i

        if bPredictive: # stop early if the predicted convergence is not sufficient
          old_errs[:-1]=old_errs[1:]
          old_errs[-1] = relerr
  
          cv_rates = old_errs[-1] / old_errs[:-1]
          for j in range(1,cv_rates.size+1):
            cv_rates[-j]=cv_rates[-j]**(1/j) # account for the intermediate iterations to obtain average convergence rates
  
          remaining_iters = np.array([maxiter-i+j for j in range(0,cv_rates.size)])
          predicted_errors = old_errs[1:]*cv_rates**remaining_iters

          self.logger.log(FIXEDPOINT, "\tcv_rates={}".format(array2exp(cv_rates)))
          self.logger.log(FIXEDPOINT, "\tpredicted_errors={}".format(array2exp(predicted_errors)))
          if i>2: # wait until at least 3 iterations have been performed (to avoid misjudging potential convergence due to oscillations of the error...)
            if np.nanmin(cv_rates)>1.: # best case diverges
              break
            if np.nanmin(predicted_errors)>1.: # best case will not match the error tolerance
              break

        p0 = p
    msg = "Failed to converge after %d iterations, value is %s" % (i, p)
    self.logger.critical(msg)
    raise WRnonConvergence(msg)


class AitkenScalarSolver(FixedPointSolver):
  def __init__(self, logger):
    super().__init__(logger)
    self.method='del2'
