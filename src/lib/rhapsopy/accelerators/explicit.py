#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:21:09 2023

@author: lfrancoi
"""
from src.lib.rhapsopy.coupling import FIXEDPOINT
from src.lib.rhapsopy.accelerators.base import BaseFixedPointSolver

class ExplicitSolver(BaseFixedPointSolver):
  def __init__(self, logger, *args):
    super().__init__(logger)

  def solve(self, fun, x0, **kwargs):
    # for i in range(maxiter): # possibility to do more than 1 iteration, even though this should not be used
    self.logger.log(FIXEDPOINT,'Explicit solver: performing single iteration')
    return fun(x0), 1
    # print('/!\ performing double explicit iteration')
    # return fun(fun(x0)), 1
