#!/usr/bin/env python3
# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:23:30 2023

@author: lfrancoi
"""
import numpy as np
from src.lib.rhapsopy.coupling import FIXEDPOINT, WRnonConvergence

class BaseFixedPointSolver():
  def __init__(self, logger):
    if logger is None:
      print('logger is None')
      import logging
      logger = logging.getLogger('BaseFixedPointSolver')
    
    self.logger = logger

  def solve(self, fun, x0, ftol, rtol, maxiter):
    raise NotImplementedError()

  def compute_error(self, x1, x2, rtol):
    # return  np.linalg.norm( (x1-x2) / ( rtol + rtol*abs(x2) ) )   /   np.sqrt(x1.size) # new version
    # return  np.max( abs(x1-x2) / ( rtol + rtol*abs(x2) ) )
    
    # old version 
    from scipy._lib._util import _asarray_validated, _lazywhere
    def _relerr(actual, desired):
        return (actual - desired) / desired
    relerr = _lazywhere(x2 != 0, (x1, x2), f=_relerr, fillvalue=x1) / rtol # same as Scipy's fixed point algorithm
    return np.max(np.abs(relerr))

  def compute_error_norm(self, error):
    return np.linalg.norm(error) / np.sqrt(error.size)

