#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:19:37 2023

@author: lfrancoi
"""
import numpy as np
from src.lib.rhapsopy.coupling import FIXEDPOINT, WRnonConvergence
from src.lib.rhapsopy.accelerators.base import BaseFixedPointSolver
import scipy


class AitkenUnderrelaxationSolver(BaseFixedPointSolver):
  def __init__(self, logger, omega0=0.5):
    super().__init__(logger)
    self.omega0 = omega0

  def solve(self, fun, x0, ftol, rtol, maxiter):
    x      = [x0.copy()]
    xtilde = [fun(x0)]
    omega  = [self.omega0]

    # 1st iteration
    it = 1
    x.append( omega[-1]*xtilde[-1]+(1-omega[-1])*x[-1] )
    xtilde.append( fun(x[-1]) )
    error = self.compute_error(x[-1], x[-2], rtol=rtol)
    self.logger.log(FIXEDPOINT, f'\tit={it}, omeg={omega[-1]:.2e}, err={error:.2e}')

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

      error = self.compute_error(x[-1], x[-2], rtol=rtol)
      self.logger.log(FIXEDPOINT, f'\tit={it}, omeg={omega[-1]:.2e}, err={error:.2e}')
      if error < 1.:
        bConverged = True
        break

      xtilde.append( fun(x_kp1) )
      if not bConverged and (it>maxiter):
        msg = 'Aitken underrelaxation did not converge'
        self.logger.log(FIXEDPOINT, msg)
        raise WRnonConvergence(msg)
    return x[-1], it
