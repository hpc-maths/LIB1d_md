#!/usr/bin/env python3
# Copyright 2022 LIB1D_MD TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:18:32 2023

@author: lfrancoi
"""
import numpy as np
from src.lib.rhapsopy.coupling import FIXEDPOINT, WRnonConvergence
from src.lib.rhapsopy.accelerators.base import BaseFixedPointSolver
import scipy

class IQNSolver(BaseFixedPointSolver):
  def __init__(self, logger, omega=0.5):
    super().__init__(logger)
    self.W = []
    self.V = []
    self.J = None
    self.max_used_steps=None
    self.omega=0.5

  def solve(self, fun, x0, ftol, rtol, maxiter):
    x      = [x0.copy()]
    xtilde = [fun(x0)]
    R      = [xtilde[0] - x[0]]

    # 1st iteration
    it = 0
    x.append( x0 + self.omega0*R[-1] )
    error = self.compute_error(x[-1], x[-2], rtol=rtol)
    # print(f'\tit={it}, err={error:.2e}')

    bConverged = False
    # ak denotes a^k (superscript), while a_k denotes an indice
    while not bConverged:
      it+=1
      xk  = x[-1]
      xtk = fun(xk)
      xtilde.append( xtk )

      # early error check
      error = self.compute_error(xtk, xk, rtol=rtol)
      if error < 1.:
        bConverged=True
        break

      # form residual vector
      Rk = xtk - xk
      R.append( Rk )
      if len(R)>4:
        R_k      =      np.array(R[-4:]).T
        xt_array = np.array(xtilde[-4:]).T

        V_k = np.diff(R_k,  axis=1)
        W_k = np.diff(xt_array, axis=1)
        # if V_k.ndim==1: # only 1 coupling variable
        #   V_k = V_k[np.newaxis,:]
        #   W_k = W_k[np.newaxis,:]

        if 0:
          # QR decomposition of V_k
          Q,U = scipy.linalg.qr(V_k, overwrite_a=False, lwork=None, mode='full',
                          pivoting=False, check_finite=True)
          # solve
          print('shape(V_k)=', V_k.shape)
          print('shape(Q)=',   Q.shape)
          print('shape(U)=',   U.shape)
          alpha = scipy.linalg.solve(a=U, b=-Q.T @ R_k, check_finite=True)
          dxtk = W_k @ alpha
        else:
          # direct pseudo-inverse, solve V_k alpha = - R_k
          alpha = - np.linalg.pinv( V_k ) @ R_k
          dxtk = W_k @ alpha

        x_kp1 = xtk + dxtk
        x.append( x_kp1.copy() )

      else: # underrelaxed fixed-point
        # x.append( x[-1] + omega0*R[-1] )
        x.append( xk    + self.omega0*(xtk - xk) )

      error = self.compute_error(x[-1], x[-2], rtol=rtol)
      # print(f'\tit={it}, err={error:.2e}')
      if error<1e-15: # issue at first iteration ?
        print('weird')
        continue
      if error < 1.:
        bConverged = True
      if not bConverged and (it>maxiter):
        raise WRnonConvergence('IQN did not converge')
    return x[-1], it

