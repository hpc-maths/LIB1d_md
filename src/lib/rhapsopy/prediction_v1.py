#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:29:15 2021

This library implements a polynomial predictor class, which allows for data to be sampled
dynamically and a fitting polynomial to be evaluated.

This library is a building block for the co-simulation approach to coupled problems.

@author: lfrancoi
"""
import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger("md_sim.prediction")

SUCCESS_THRESHOLD=3 # number of successful steps required before considering an order increase

def compute_divided_diff_coef(x, y):
    """ Computes the divided difference coefficients required for the
    evaluation of NEwton-type polynomials. """
    n = x.size
    coef = y.copy()
    try:
      for i in range(1,n):
          coef[i:] = (coef[i:] - coef[i-1:-1])/(x[i:] - x[:-i])
    except FloatingPointError as e:
      raise e
    return coef

def newton_interp(xi, yi, x):
    """ CoOmputes the polynomial coefficients and evaluates the polynomial
    at the inpmut points x """
    coef = compute_divided_diff_coef(xi, yi)
    return newton_interp_coef(xi, x, coef)

def newton_interp_coef(xi, x, coef):
    """ Evaluates the polynomial at input x, with coefficients already computed
    at the sampling points xi. """
    n = xi.size
    p = np.zeros(x.size)
    for i in range(n-1,0,-1):
        p = (coef[i]+p) * (x-xi[i-1])
    p = p +  coef[0]
    return p


class predicteur:
  """ This class represent a polynomial interpolation of a series of data points.
      The predictor can approximate in a ppolynomial fashion the evolution of one or more variables.
      The predictors uses up to NMAX sampling points.
      In the build-up phase, fewer points may be available, hence the number of points used for the polynomial interpolation is dynamically reduced.
      The order of the predictor is the order of the interpolation error with respect to the grid spacing.
      For instance, if the predictor only uses a single sampling point (x0,y0), the polynomial representation is simply a constant extrapolation.
      Hence, the error is of order 1.

            y(x0+dx)-yp(x0+dx) = y(x0+dx) - y0 =  y'(x0)*dx + ... ~ O(dx)
      with y denoting the exact function, and yp its polynomial approximation.

      More generally, the degree of the polynomial prediction is equal to N-1, N being the number of sampling points,
      and the order of accuracy is N.

      """
  def __init__(self,NMAX):
    # Basic properties required for polynomial definition
    self.N=0 # current order
    self.Nacquired = 0 # current max order = number of data points gathered
    self.ndata=0 # number of data components
    self.NMAX = NMAX #5 # maximum order of the prediction
    self.x = None # data abscissa
    self.y = None # data values
    self.coef = None # coefficients of the Newton polynomial

    # Control properties
    self.needRefresh = True # whether or not the Newton coefficients should be reocmputed before the next evaluation (e.g. order change, new data)

    # Properties specific for time step and order adaptation
    self.atol = None
    self.rtol = None # absolute and relative tolerances for the step adaptation based on the predictor's accuracy
    self.nsuccess_with_current_order=0 # number of successful steps with the current order

  def changeNmax(self, NMAX):
    """ Changes the maximum number of sampling points """
    oldX=np.copy(self.x)
    oldY=np.copy(self.y)
    oldcoef=np.copy(self.coef)

    self.x = np.zeros((NMAX,))
    self.y = np.zeros((self.ndata,NMAX))
    self.coef = np.zeros((self.ndata,NMAX))

    imax = min(self.NMAX,NMAX)
    self.x[:imax] = oldX[:imax]
    self.y[:,:imax] = oldY[:,:imax]
    self.coef[:,:imax] = oldcoef[:,:imax]

    self.NMAX = NMAX

  def clone(self):
    """ Returns a copy of the predictor """
    pred = predicteur(NMAX=self.NMAX)
    pred.N=self.N
    pred.Nacquired=self.Nacquired
    pred.ndata=self.ndata
    pred.nsuccess_with_current_order = self.nsuccess_with_current_order

    pred.needRefresh =self.needRefresh
    pred.x =np.copy(self.x)
    pred.y =np.copy(self.y)
    pred.coef =np.copy(self.coef)

    pred.atol =self.atol
    pred.rtol =self.rtol
    return pred

  def getX(self):
    """ Returns only the used portion of the sampling points abscissae """
    return self.x[:self.N]

  def getY(self):
    """ Returns only the used portion of the sampling points values """
    return self.y[:,:self.N]

  def getNacquired(self):
    """ Returns the number of sampling points stored """
    return self.Nacquired

  def getN(self):
    """ Returns the number of sampling points activated (i.e. the order of the approximation) """
    return self.N

  def getCoef(self):
    """ Returns the polynomial coefficients """
    if self.needRefresh:
      self.updateCoef()
      self.needRefresh=False
    return self.coef[:, :self.N]

  def setCurrentN(self,N):
    """ Sets the new order for the polynomial extrapolation (i.e. number of past points used) """
    if (N>self.NMAX):
        raise Exception( f'The required order (N={N}) should not be larger than NMAX={self.NMAX}' )
    if (N>self.Nacquired):
        raise Exception( f'The required order (N={N}) should not be larger than Nacquired={self.Nacquired}' )
    # If the order is lowered, we also discard old data points that won't be used,
    # this is especially useful if the order reduction is caused by a discontinuity.
    # if (N < self.N):
    #     self.Nacquired = N
    #     #self.Nacquired = min(N+1, self.Nacquired)
    #     #self.Nacquired = min(self.Nacquired, self.NMAX)

    # Set the new order
    if (self.N != N): # order is indeed changed
        self.nsuccess_with_current_order = 0 # reset the success counter
    self.N = N
    self.needRefresh = True


  def setTolerances(self, atol, rtol):
    """ Specifies the tolerance used for error estimation """
    self.atol = atol
    self.rtol = rtol

  def decreaseN(self):
    # if self.N<self.NMAX: # otherwise, the previous point has already been dropped and we would reduce the order...
    #   self.N = self.N-1
    #   # BUT we do NOT discard the oldest point, so we can reuse it later
    #   assert self.N>=0
    self.N = min(0, self.N-1)

  def increaseN(self):
    # if self.N<self.Nacquired:
    self.N = min(self.Nacquired, self.N+1)

  def reset(self):
    self.__init__(NMAX=self.NMAX)

  def replaceLastPoint_single(self,x,y):
    """ Replaces the last sampling point, useful for iterative co-simulation """
    y_vec = np.array([y])
    self.replaceLastPoint_vec(x, y_vec)

  def replaceLastPoint_vec(self,x,y):
    """ Replaces the last sampling point, useful for iterative co-simulation """
    # add the new values, reaplcing the first ont
    self.x[0] = x
    self.y[:,0] = y[:]
    self.needRefresh = True
    self.coef[:,:] = 0. # unnecessary safety

  def discard_oldest_point(self):
    """ Drop the oldest point """
    self.N = self.N-1
    self.Nacquired = self.Nacquired -1
    assert self.N>0, 'all sampling points have been discarded'
    self.needRefresh = True
    self.coef[:,:] = np.nan # unnecessary safety


  def appendData_single(self,x,y,bPerformChecks=False):
    """ this routine allows to add a new data point (at a new time point) """
    y_vec = np.array((y,))
    self.appendData_vec(x,y_vec,bPerformChecks=bPerformChecks)


  def appendData_vec(self,x,y, bPerformChecks=False):
    """ this routine allows to add a set of new data points (at a new time point) """
    ndata = y.shape[0]

    if self.x is None: # allocate storage for the data
        self.x = np.zeros((self.NMAX,))
        self.y = np.zeros((ndata, self.NMAX))
        self.coef = np.zeros((ndata, self.NMAX))
        self.ndata = ndata
    else:
      if bPerformChecks:
        # sanity checks
        if (self.x[0] >= x):
            raise Exception('The new time point is not greater than the previous one...')
        if (self.ndata != ndata):
            raise Exception(f'The new data set has more components ({ndata}) than the previous one ({self.ndata})')
        # check that time steps do not increase too much
        if self.Nacquired>0:
            tmp = np.diff(self.x[:self.Nacquired])
            if not np.size(np.unique(np.sign(tmp)))==1:
                print('x=',self.x)
                print('y=',self.y)
                print('N=',self.N)
                print('Nacquired=',self.Nacquired)
                import pdb; pdb.set_trace()
                raise Exception('Predictor time points are not monotonous (1)')
            if np.min(tmp)/np.max(tmp) < 0:
                raise Exception('Predictor time points are not monotonous (2)')
            if np.min(tmp)/np.max(tmp) < 0.1:
                raise Exception('Gaps between successive predictor time points evolve too quickly')

    # "move" the previous points
    self.x[  1:self.NMAX]   = self.x[  :self.NMAX-1]
    self.y[:,1:self.NMAX]   = self.y[:,:self.NMAX-1]
    # --> the previous oldest point is automatically dropped

    # add the new values
    self.x[0] = x
    self.y[:,0] = y[:]

    # unnecessary safety
    self.coef[:,:] = 0.

    # We have added one point, therefore we can increase the order  of the polynomial
    self.Nacquired = min(self.NMAX, self.Nacquired+1)
    self.N = min(self.Nacquired, self.N+1)

    self.needRefresh = True # coeffs need to be updated at the next call

    # TODO: specific function to record the sequence of successful time steps for co-simulation
    # self.nsuccess_with_current_order = self.nsuccess_with_current_order + 1

  def updateCoef(self):
    """ Computes the new coefficients based on the current data and N """
    for i in range(0, self.ndata):
      self.coef[i,:self.N] = compute_divided_diff_coef( self.x[:self.N], self.y[i,:self.N] )
    self.needRefresh = False


  def reportDiscontinuity(self):
    """ A discontinuity in the predicted model has been spotted, therefore we
        discard all points except the newest one """
    self.N = 1
    self.Nacquired = 1
    self.needRefresh = True
    self.x[1:] = np.nan
    self.y[:,1:] = np.nan

  def transmitTimeAndCoef(self):
    """ Returns the sampling points abscissae and values """
    return self.x[:self.N],  self.coef[:,:self.N]

  def eval_single(self,x,allow_outside=False):
    """ Same eval but for a single time point --> interface to the vectorized function """
    x_vec = np.array(x)
    v_vec = self.eval_vector( x_vec, allow_outside=allow_outside )
    v = v_vec[:,0]
    return v

  def eval_vector(self,x,allow_outside=False):
    """ Evaluate the current polynomial interpolation at the given array of abscissas.
        If allow_outside is False, an error is raised if the evaluation abscissa x is
        outside of the interval covered by the sampling point abscissae """
    if not isinstance(x,np.ndarray):
      x=np.array(x)
    if self.x is None:
        raise Exception('The polynomial predictor has not been initialised with any data yet, cannot evaluate')

    v = np.zeros( ( self.ndata, x.size ) )

    if (self.needRefresh):
      # refresh the coefficients
      self.updateCoef()

    # if not allow_outside: # verify that we are not calling outside of the time domain sampled (sanity check for integration)
    # # does not work for Runge-Kutta scheme which have some c_i>1...
    #     try:
    #         if self.x.size>1:
    #           if self.x[1]<self.x[0]: # integration happens in the direction t>0
    #             assert np.all(x<=self.x[0])
    #             assert np.any(x>=self.x[1])
    #           else: # time decreases
    #             assert np.all(x>=self.x[0])
    #             assert np.any(x<=self.x[1])
    #     except Exception as e:
    #         print(e)
    #         print('self.x=',self.x)
    #         print('x=',x)
    #         raise e
    for i in range(self.ndata):
      v[i,:] = newton_interp_coef(xi=self.x[:self.N], x=x, coef=self.coef[i,:self.N])
      # v[i,:] = newton_interp( xi=self.x[:self.N+1], yi=self.y[i,:self.N+1], x=x ) # forces the coefficients to be evaluated anew
      # debug --> l'adaptation d'ordre marche bien mieux avec ça... TODO: POURQUOI ???
      # C'est censé être équivalent... À moins que les coefs n'ait pas été recalculés ?? --> si ?
    return v

  def eval_error_single(self, ref_value, time, order=None):
    """ Computes the estimated prediction error on self predictor's extrapolated variable,
        compared to the true value (obtained by numerical coupling for example) "ref_value", at time "time". """
    ref_value_vec = np.array(ref_value)
    error_vec = self.eval_error_vector( ref_value_vec, time, order=order )
    return error_vec[0]

  def eval_error_vector(self, ref_value, time, order=None):
    """ Computes the estimated prediction error on self predictor's extrapolated variable,
        compared to the true value (obtained by numerical coupling for example) "ref_value", at time "time".
        The argument order specifies which prediction order should be used to estimate the error.
        It is useful to study the impact of the prediction order and decide wheter the current order may be changed """
    if (self.ndata != ref_value.size):
        raise Exception(f'ref_value is of size {ref_value.size}, whereas the predictor is set for {self.ndata} variables')

    backup_N = self.N
    bOrderChanged = False
    if not (order is None):
        # assert self.N>=order-1, f'required order ({order}) cannot be satisfied with the current number of sample points ({self.N})'
        self.N = order
        self.needRefresh = True
        bOrderChanged = True

    # Evaluate error (with the chosen prediction order)
    error = abs( self.eval_vector(time) - ref_value ) / ( self.atol + self.rtol * abs(ref_value) )

    if bOrderChanged:
      self.N = backup_N # restore original order
      self.needRefresh = True
      # TODO: or simply backup coefficients
    return error

  def eval_optimal_timestep_single(self, ref_value, time, dt=1, order=None):
    ref_value_vec = np.array( ref_value )
    dt_opt = self.eval_optimal_timestep_vector( ref_value_vec, time, dt, order )
    return dt_opt

  def eval_optimal_timestep_vector(self, ref_value, time, dt=1, order=None):
    """ Compute the optimal time step with the current order (or optionally assuming a different order)
        self is most coherent when the order is the same as the one used for the coupling step that has just been performed, self way
        the predicted evolution is the same in the error estimation as it was in the split-integrated models. """

    if order is None:
      order = self.N
    factor_opts = np.zeros( self.ndata )
    # import pdb; pdb.set_trace()
    errors = abs( self.eval_error_vector( ref_value, time, order=order ) )
    if not all(errors>=0):
        raise Exception('issue')
    errors = np.maximum( 1e-15, errors )

    for i in range(self.ndata):
        factor_opts[i] = (1./errors[i])**(1/order) #(1./(1.+order))
        # factor by which the time step can be increased while still satisfying the error constraints
    return dt * np.min( factor_opts )

  def suggest_order_error_based(self, ref_value, time):
    # The order of the predictor is adjusted such that the allowed time step is largest for a given relative error level
    # TODO: merge self with eval_timestep to suggest both orders and time step for the next step
    # self stragey is inspired by the strategies described by Petzold, Shampine and others for multistep methods, however
    # here the frameworik is slightly different: the predictor does not solve and ODE, and here the new order will be selected
    # by seeing if, had it been selected one step before, it would have led to a greater optimal time step

    # TODO: comment gérer différentes variables à la fois avec un seul prédicteur (i.e. self.ndata > 1) ???
    # --> pour le pas de temps, on prend le plus faible, mais pour l'ordre ?
    if (self.ndata>1):
        raise Exception('Predictors are only compatible with a single variable per predictor at the moment')

    # test all orders
    order_min = 1
    order_max = min(self.Nacquired, self.N+1) # the order can only be increased by 1 (but may be decreased arbitrarily)
    orders = np.array(range(order_min,order_max+1))

    factor_opts = np.zeros((self.ndata, orders.size))
    errors      = np.zeros((self.ndata, orders.size))

    for iord,order in enumerate(orders):
        # From order 1 to the maximum order achievable with the acquired data points
        factor_opts[:,iord] = self.eval_optimal_timestep_single(ref_value=ref_value,
                                                        time=time,
                                                        order=order)

        # only for debug
        errors[:,iord] = self.eval_error_single(ref_value=ref_value, time=time, order=order)

    print('**** Suggesting order (error based)')
    for i in orders:
      print('****  orders     = ', orders)
      print('****  errors     = ', errors)
      print('****  dt_opts/dt = ', factor_opts)

    # Choose the new order as the one that allows for the largest time step
    order = orders[ np.argmax(factor_opts[0,:]) ]

    print('**** new order = ', order)
    if (order>self.N): # check that we do not increase the order too rapidly
      if (self.nsuccess_with_current_order < SUCCESS_THRESHOLD ):
        print('**** successful steps with previous order = ', self.nsuccess_with_current_order, ' < ', SUCCESS_THRESHOLD)
        print('     --> order is not increased yet')
        order = self.N+1
    print('')
    return order

if __name__=='__main__':

    NMAX = 5
    print('Testing Newton interpolation')
    for npts in range(1,5+1):
      xi = np.linspace(0,1, npts) + 1/(2*npts)*np.random.rand(npts) ##np.array([0,1,3])
      fun = lambda x: 10 + x**2 + 3*x + 0.1*x**(npts-1)
      # fun = lambda x: 100 + x**(npts-1)
      # fun = lambda x: np.arctan(x)
      yi = fun(xi)

      x = np.linspace(-1,2,100)
      y1 = newton_interp(xi=xi, yi=yi, x=x)

      poly = np.polyfit(x=xi, y=yi, deg=npts-1)
      y2 = np.polyval(p=poly, x=x)

      # also test the predictor class below
      pred = predicteur(NMAX=NMAX)
      for i in range(xi.size):
        pred.appendData_single(x=xi[i], y=yi[i], bPerformChecks=False)

      y3 = pred.eval_vector(x)[0,:]

      yref = fun(x)

      plt.figure()
      plt.plot(x, y1, label='Newton', linewidth=3)
      plt.plot(x, y2, label='polyfit', linewidth=3)
      plt.plot(x, y3, label='predictor', linewidth=3)
      plt.plot(x, yref, label='ref', linestyle='-.')
      plt.plot(xi, yi, label=None, linestyle='', marker='+')
      plt.legend()
      plt.grid()
      plt.xlabel('x')
      plt.title(f'Predictions for npts={npts}')

      plt.figure()
      plt.semilogy(x, abs( (y1-yref)/yref ), label='Newton')
      plt.semilogy(x, abs( (y2-yref)/yref ), label='polyfit')
      plt.semilogy(x, abs( (y3-yref)/yref ), label='predictor')
      plt.legend()
      plt.grid()
      plt.xlabel('x')
      plt.title(f'Relative errors for npts={npts}')


    # Visual test
    print('Testing predictor class')

    testfun = lambda x: np.arctan(x)
    # testfun = lambda x: x**4

    NX = 20
    xtest = np.unique( np.linspace(-2*np.pi, 2*np.pi, NX) + np.random.uniform(-1,1,NX)*0.1 )
    dx = np.mean(np.diff(xtest))

    plt.figure()
    plt.plot(xtest, testfun(xtest), marker='+')
    plt.grid()

    predictor = predicteur(NMAX=NMAX)

    # on teste le prédicteur sur chaque valeur discrète
    for i in range(len(xtest)):
      current_x = xtest[i]
      current_y = testfun(xtest[i])
      # if i<predictor.NMAX:
      predictor.appendData_single( x=current_x, y=current_y )
      # else:
      #   predictor.replaceLastPoint_single( x=current_x, y=current_y )

      # evaluate on a small interval near the current point
      # xpred = np.linspace(np.min(predictor.x), xtest[i]+3*dx, 100)
      xpred = np.linspace(xtest[0], xtest[-1], 100)
      ypred = predictor.eval_vector( x=xpred )[0,:]

      plt.figure()
      plt.plot(xtest, testfun(xtest), marker='+')
      plt.plot(current_x, current_y, marker='o', color='r', label=None)
      plt.plot(predictor.x, predictor.y[0,:], marker='*', linestyle='', color='r', label=None)
      plt.plot(xpred, ypred, marker=None, label='prediction', linestyle='--')
      # plt.xlim(xtest[i]-4*dx, xtest[i]+4*dx)
      # plt.ylim( np.min(ypred), np.max(ypred))
      # plt.ylim(-1e4,1e4)
      plt.ylim(-1.5,1.5)
      plt.title(f'N={predictor.N}')

      # manually enforce the order increase
      # if (predictor.Nacquired < predictor.NMAX): # we are still in the build-up phase, i.e. the nubmer of data points is still increasing
      #   predictor.N = predictor.N + 1 # increase order anyway: TODO: make self better



