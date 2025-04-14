#!/usr/bin/env python3
# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

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

class Broyden1Solver(BaseFixedPointSolver):
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
      sol = scipy.optimize.broyden1(F=resfun, xin=x0, iter=None,
                                    alpha=None, reduction_method='restart',
                                    max_rank=None, verbose=True, maxiter=None,
                                    f_tol=1e-12, f_rtol=None, x_tol=None, x_rtol=max(1e-6,rtol/10),
                                    tol_norm=None, line_search='armijo', callback=None)
    except NoConvergence as e:
      print(e)
      msg = "Broyden2 Failed to converge after {} calls".format(ncalls)
      self.logger.critical(msg)
      raise WRnonConvergence(msg)
      
    except (OverflowError, ValueError) as e:
      print(e)
      msg = "Broyden2 internal error after {} calls ({})".format(ncalls,e)
      self.logger.critical(msg)
      raise ExceptionWhichMayDisappearWhenLoweringDeltaT(msg)
      
    return sol, ncalls

class Broyden2Solver(BaseFixedPointSolver):
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
      sol = scipy.optimize.broyden2(F=resfun, xin=x0, iter=None,
                                    alpha=None, reduction_method='restart',
                                    max_rank=None, verbose=True, maxiter=None,
                                    f_tol=1e-12, f_rtol=None, x_tol=None, x_rtol=max(1e-6,rtol/10),
                                    tol_norm=None, line_search='armijo', callback=None)
    except NoConvergence as e:
      print(e)
      msg = "Broyden2 Failed to converge after {} calls".format(ncalls)
      self.logger.critical(msg)
      raise WRnonConvergence(msg)
      
    except (OverflowError, ValueError) as e:
      print(e)
      msg = "Broyden2 internal error after {} calls ({})".format(ncalls,e)
      self.logger.critical(msg)
      raise ExceptionWhichMayDisappearWhenLoweringDeltaT(msg)
      
    return sol, ncalls