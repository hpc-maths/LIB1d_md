#!/usr/bin/env python3
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
    return  np.linalg.norm( (x1-x2) / ( rtol + rtol*abs(x2) ) )   /   np.sqrt(x1.size)

  def compute_error_norm(self, error):
    return np.linalg.norm(error) / np.sqrt(error.size)

