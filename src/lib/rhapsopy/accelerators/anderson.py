#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:21:09 2023

@author: lfrancoi
"""
import numpy as np
from src.lib.rhapsopy.coupling import FIXEDPOINT, WRnonConvergence
from src.lib.rhapsopy.accelerators.base import BaseFixedPointSolver
import scipy

class AndersonSolver(BaseFixedPointSolver):
  def __init__(self, logger, method='del2'):
    super().__init__(logger)
    self.method=method

  def solve(self, fun, x0, ftol, rtol, maxiter):
    return scipy.optimize.anderson(F=lambda x: x-fun(x),
                                    xin=x0.copy(),
                                    alpha=None, w0=0.01, M=5, verbose=False,
                                    maxiter=self.NITER_MAX, f_tol=ftol, f_rtol=None,
                                    x_tol=rtol, x_rtol=rtol,
                                    tol_norm=None, line_search='armijo'), np.nan # pas moyen d'avoir n_iter ?
