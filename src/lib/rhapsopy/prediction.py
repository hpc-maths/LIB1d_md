#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:29:15 2021

This library implements a polynomial predictor class, which allows for data to be sampled
dynamically and a fitting polynomial to be evaluated.

This library is a building block for the code coupling approach to coupled problems.

@author: lfrancoi
"""
import matplotlib.pyplot as plt
import numpy as np
import logging
import scipy
import scipy.integrate

logger = logging.getLogger("rhapsopy.prediction")

SUCCESS_THRESHOLD=3 # number of successful steps required before considering an order increase
EXTRA = 3

def compute_divided_diff_coef(x, y):
    """ Computes the divided difference coefficients required for the
    evaluation of Newton-type polynomials. """
    n = x.size
    coef = y.copy()
    # try:
    for i in range(1,n):
        coef[i:] = (coef[i:] - coef[i-1:-1])/(x[i:] - x[:-i])
    # except FloatingPointError as e:
    #   raise e
    return coef

def newton_interp(xi, yi, x):
    """ Computes the polynomial coefficients and evaluates the polynomial
    at the inpmut points x """
    coef = compute_divided_diff_coef(xi, yi)
    return newton_interp_coef(xi, x, coef)

def newton_interp_coef(xi, x, coef):
    """ Evaluates the polynomial at input x, with coefficients already computed
    at the sampling points xi. """
    n = xi.size
    if isinstance(x, np.ndarray):
      lx = x.size
    else:
      lx = 1
    p = np.zeros(lx)
    for i in range(n-1,0,-1):
        p = (coef[i]+p) * (x-xi[i-1])
    p = p +  coef[0]
    return p

def newton_interp_coef2(xi,x,coef):
  try:
    return newton_interp_coef(xi,x,coef)
  except IndexError:
    import pdb; pdb.set_trace();
    return newton_interp_coef(xi,x,coef)

def least_square_eval(poly, x):
    return np.polyval(poly,x)

def least_square_fit_coeff(xi, yi):
    raise NotImplementedError()

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
    self.N = 0 # current order
    self.Nacquired = 0 # current max order = number of data points gathered
    self.ndata = 0 # number of data components
    self.NMAX = NMAX # maximum order of the prediction
    self.NSTORE = NMAX+EXTRA # storage size
    self._x = None # data abscissa
    self._y = None # data values
    self._coef = None # coefficients of the Newton polynomial

    # Control properties
    self.needRefresh = True # whether or not the Newton coefficients should be reocmputed before the next evaluation (e.g. order change, new data)

    # Properties specific for time step and order adaptation
    self.atol = None
    self.rtol = None # absolute and relative tolerances for the step adaptation based on the predictor's accuracy
    self.nsuccess_with_current_order = 0 # number of successful steps with the current order
    self.error_method = -1 # method to evaluate the error

  def _init_storage(self):
    self._x    = np.empty((self.NSTORE,))*np.nan
    self._y    = np.empty((self.ndata, self.NSTORE))*np.nan
    self._coef = np.empty((self.ndata, self.NSTORE))*np.nan

  def __str__(self):
    return f"""Predictor with N={self.N}, NMAX={self.NMAX}, Nacquired={self.Nacquired}, NSTORE={self.NSTORE}, ndata={self.ndata},  nsuccess_with_current_order={self.nsuccess_with_current_order}"""

  def changeNmax(self, NMAX):
    """ Changes the maximum number of sampling points """

    new_NSTORE = NMAX + EXTRA
    if new_NSTORE > self.NSTORE:
      oldX = np.copy(self._x)
      oldY = np.copy(self._y)
      oldcoef = np.copy(self._coef)

      # increase storage size
      self._init_storage()

      # copy over old data
      imax = self.NSTORE #min(self.NMAX+EXTRA, NMAX+EXTRA)
      self._x[:imax] = oldX[:imax]
      self._y[:,:imax] = oldY[:,:imax]
      self._coef[:,:imax] = oldcoef[:,:imax]

    else: # simply cut old data space
      self._x    = self._x[:new_NSTORE]
      self._y    = self._y[:,:new_NSTORE]
      self._coef = self._coef[:,:new_NSTORE]

    self.NSTORE = new_NSTORE
    self.NMAX = NMAX

  def clone(self):
    """ Returns a copy of the predictor """
    pred = predicteur(NMAX=self.NMAX)

    pred.N = self.N
    pred.Nacquired = self.Nacquired
    pred.ndata = self.ndata
    pred.error_method = self.error_method
    pred.nsuccess_with_current_order = self.nsuccess_with_current_order

    pred.needRefresh = self.needRefresh
    pred._x    = np.copy(self._x)
    pred._y    = np.copy(self._y)
    pred._coef = np.copy(self._coef)

    pred.atol = self.atol
    pred.rtol = self.rtol
    return pred

  @property
  def coef(self):
    """ Returns the polynomial coefficients """
    if self.needRefresh:
      self._updateCoef()
    return self._coef[:, :self.Nacquired]

  @property
  def x(self):
    """ Returns the sampling abscissae """
    return self._x[:self.Nacquired]

  @property
  def y(self):
    """ Returns the sampling abscissae """
    return self._y[:,:self.Nacquired]

  def _updateCoef(self):
    """ Computes the new coefficients based on the current data and N """
    if self._x is None:
        raise Exception('The polynomial predictor has not been initialised with any data yet, cannot evaluate')
    for i in range(0, self.ndata):
      self._coef[i,:self.Nacquired] = compute_divided_diff_coef( self._x[:self.Nacquired], self._y[i,:self.Nacquired] )
    self.needRefresh = False

  def setCurrentN(self,N):
    """ Sets the new order for the polynomial extrapolation (i.e. number of past points used) """
    # if (N > self.NMAX):
    #     raise Exception( f'The required order (N={N}) should not be larger than NMAX={self.NMAX}' )
    if (N > self.Nacquired):
        raise Exception( f'The required order (N={N}) should not be larger than the number of acquired points ({self.Nacquired})')
    # If the order is lowered, we also discard old data points that won't be used,
    # this is especially useful if the order reduction is caused by a discontinuity.
    # if (N < self.N):
    #     self.Nacquired = N
    #     #self.Nacquired = min(N+1, self.Nacquired)
    #     #self.Nacquired = min(self.Nacquired, self.NMAX)
    # print('changing N from', self.N, 'to ', N)
    assert N > 0, 'N cannot be lower than 1'
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
    self.N = min(1, self.N-1)

  def increaseN(self):
    self.N = min(self.Nacquired, self.N+1)

  def reset(self):
    atol, rtol = self.atol, self.rtol
    error_method = self.error_method
    self.__init__(NMAX=self.NMAX)
    self.atol, self.rtol = atol, rtol # restore tolerances
    self.error_method = error_method

  def replaceLastPoint(self,x,y):
    """ Replaces the last sampling point, useful for iterative code coupling """
    if not isinstance(y,np.ndarray):
      y = np.array((y,))
    # add the new values, reaplcing the first ont
    self._x[0] = x
    self._y[:,0] = y[:]
    self.needRefresh = True
    self._coef[:,:] = np.nan # unnecessary safety

  def discard_oldest_point(self):
    """ Drop the oldest point """
    self.N = self.N-1
    self.Nacquired = self.Nacquired -1
    assert self.N>0, 'all sampling points have been discarded'
    self.needRefresh = True
    self._coef[:,:] = np.nan # unnecessary safety


  def appendData(self,x,y,bPerformChecks=True,bIncreaseN=True):
    """ this routine allows to add a new data point (at a new time point) """
    if not isinstance(y,np.ndarray):
      y = np.array((y,))
    assert np.size(x)==1, 'only one time point can be added at once'

    ndata = y.shape[0]
    if self._x is None: # allocate storage for the data
        self.ndata = ndata
        self._init_storage()

    else:
      if bPerformChecks:
        # sanity checks
        if (self._x[0] >= x):
            raise Exception('The new time point is not greater than the previous one...')
        if (self.ndata != ndata):
            raise Exception(f'The new data set has more components ({ndata}) than the previous one ({self.ndata})')
        # check that time steps do not increase too much
        if self.Nacquired>1:
            tmp = np.diff(self._x[:self.Nacquired])
            if not np.size(np.unique(np.sign(tmp)))==1:
                print('x=',self._x)
                print('y=',self._y)
                print('N=',self.N)
                print('Nacquired=',self.Nacquired)
                import pdb; pdb.set_trace()
                raise Exception('Predictor time points are not monotonous (1)')
            if np.min(tmp)/np.max(tmp) < 0:
                raise Exception('Predictor time points are not monotonous (2)')
            if np.min(tmp)/np.max(tmp) < 0.1:
                raise Exception('Gaps between successive predictor time points evolve too quickly')

    # "move" the previous points
    self._x[  1:]   = self._x[  :-1]
    self._y[:,1:]   = self._y[:,:-1]
    # --> the previous oldest point is automatically dropped

    # add the new values
    self._x[0] = x
    self._y[:,0] = y[:]

    # unnecessary safety
    self._coef[:,:] = 0.

    # We have added one point, therefore we can increase the order  of the polynomial
    self.Nacquired = min(self.NSTORE, self.Nacquired+1)
    if bIncreaseN:
      self.N = min(self.NMAX, min(self.Nacquired, self.N+1) )

    self.needRefresh = True # coeffs need to be updated at the next call

    # TODO: specific function to record the sequence of successful time steps for code coupling
    # self.nsuccess_with_current_order = self.nsuccess_with_current_order + 1

  def reportDiscontinuity(self):
    """ A discontinuity in the predicted model has been spotted, therefore we
        discard all points except the newest one """
    self.N = 1
    self.Nacquired = 1
    self.needRefresh = True
    self._x[1:] = np.nan
    self._y[:,1:] = np.nan


  def evaluate(self,x,allow_outside=False):
    """ Evaluate the current polynomial interpolation at the given array of abscissas.
        If allow_outside is False, an error is raised if the evaluation abscissa x is
        outside of the interval covered by the sampling point abscissae """
    if not isinstance(x,np.ndarray):
      x = np.array(x)
      bScalarInput = True
    else:
      bScalarInput = False

    v = np.zeros( ( self.ndata, x.size ) )
    if self.needRefresh:
      self._updateCoef()

    # if not allow_outside: # verify that we are not calling outside of the time domain sampled (sanity check for integration)
    # # does not work for Runge-Kutta scheme which have some c_i>1...
    for i in range(self.ndata):
      v[i,:] = newton_interp_coef(xi=self._x[:self.N], x=x, coef=self.coef[i,:self.N])
      # v[i,:] = newton_interp( xi=self._x[:self.N+1], yi=self._y[i,:self.N+1], x=x ) # forces the coefficients to be evaluated anew
      # debug --> l'adaptation d'ordre marche bien mieux avec ça... TODO: POURQUOI ???
      # C'est censé être équivalent... À moins que les coefs n'ait pas été recalculés ?? --> si ?
    if bScalarInput:
      v = v[:,0]
    return v

  def eval_error(self, other_pred, tn, tnp1, order=None):
    """ Computes the estimated prediction error on self predictor's extrapolated variable,
        compared to the true value (obtained by numerical coupling for example) "ref_value", at time "tnp1".
        The argument order specifies which prediction order should be used to estimate the error.
        It is useful to study the impact of the prediction order and decide wheter the current order may be changed """
    # if (self.ndata != ref_value.size):
    #     raise Exception(f'ref_value is of size {ref_value.size}, whereas the predictor is set for {self.ndata} variables')
    assert tnp1>tn, 'something is strange !'
    backup_N = self.N
    bOrderChanged = False
    if not (order is None):
        # assert self.N>=order-1, f'required order ({order}) cannot be satisfied with the current number of sample points ({self.N})'
        self.N = order
        self.needRefresh = True
        bOrderChanged = True

    # Evaluate error (with the chosen prediction order)
    if self.error_method==0: # point error at last time
      ref_value = other_pred.evaluate(tnp1)
      error = abs( self.evaluate(tnp1) - ref_value ) / ( self.atol + self.rtol * abs(ref_value) )
      # TODO: for order adaptation, this will not work for implicit simulations,
      # since the new point is going to be the same anyway...
      # 2 solutions: evaluate error at mid point, or integral error (error_method=1)
      # TODO: this remains slightly unfair, since the order that has been used for the actual step will be favoured...

    elif self.error_method==1: # L2-norm over the last time step
      # out = scipy.integrate.quad(func=lambda t: (self.evaluate(t) - other_pred.evaluate(t))**2,
      #                      a=tn, b=tnp1,
      #                      full_output=0, epsabs=self.atol/20, epsrel=self.rtol/20)
      # out = scipy.integrate.quad_vec(f=lambda t: (self.evaluate(t) - other_pred.evaluate(t))**2,
      # TODO: since we are comparing polynomials, this could be heavily optimised !
      ord_norm = 2
      integrated_fun = lambda t: ( abs(self.evaluate(t) - other_pred.evaluate(t)) / ( self.atol + self.rtol * abs(other_pred.evaluate(t)) ) )**ord_norm
      if 0:
        out = scipy.integrate.quad_vec(
                f=integrated_fun,
                a=tn, b=tnp1, full_output=0,
                norm='max', epsabs=self.atol/20, epsrel=self.rtol/20)
        # print(out)
        error = out[0]
      else:
        t_eval = np.linspace(tn,tnp1,10)
        y_eval = np.vstack([integrated_fun(xx) for xx in t_eval]).T
        error = scipy.integrate.trapz(y_eval, x=t_eval)

      error = (error / (tnp1-tn))**(1/ord_norm) # erreur moyenne sur le pas de temps



    elif self.error_method==2: # upper bound of the interpolation error, see wiki or Petzold p125
      # highest_derivative = np.math.factorial(self.N) * coeff[:,self.N]
      ref_value = self.evaluate(tnp1)

      # error = self.coef[:,self.N-2]
      # for i in range(self.N-1):
      #   error *= (tnp1-self.x[i])
      # error = error / ( self.atol + self.rtol * abs(ref_value) )
      coeff = self.coef

      def get_error(t):
        error = np.copy(coeff[:,self.N-1])
        for i in range(self.N):
          error *= (t-self.x[i])
        return error

      def integrated_fun(t):
        return ( get_error(t) / ( self.atol + self.rtol * abs(self.evaluate(t)) ) )**2

      # integrated_fun(tnp1) # test call

      # integrate over last sampling interval
      if 0:
        out = scipy.integrate.quad_vec(
          f = integrated_fun,
          a=tn, b=tnp1, full_output=0,
          norm='max', epsabs=self.atol/20, epsrel=self.rtol/20)
        error = np.sqrt(out[0] / (tnp1-tn)) # erreur quadratique moyenne sur le pas de temps
      else:
        t_eval = np.linspace(tn,tnp1,10)
        # import pdb; pdb.set_trace()
        y_eval = np.vstack([integrated_fun(xx) for xx in t_eval]).T
        error = scipy.integrate.trapz(y_eval, x=t_eval)
        error = np.sqrt(error / (tnp1-tn)) # erreur quadratique moyenne sur le pas de temps

      logger.log(100, f"   tn={tn}, tnp1={tnp1}")
      logger.log(100,  "   x="+str(self.x[:self.N]) )
      logger.log(100,  "   error2="+str(error.tolist()) )

    else:
      raise Exception(f'error method {self.error_method} unknown')

    if bOrderChanged:
      self.N = backup_N # restore original order
      self.needRefresh = True
      # TODO: or simply backup coefficients
      # TODO: coeff only to be updated if values are changed...
    if not np.all(error>=0):
      import pdb; pdb.set_trace()
      raise Exception('error is not positive')

    return error


  def eval_optimal_timestep(self, other_pred, tn, tnp1, order=None):
    """ Compute the optimal time step with the current order (or optionally assuming a different order)
        self is most coherent when the order is the same as the one used for the coupling step that has just been performed, self way
        the predicted evolution is the same in the error estimation as it was in the split-integrated models. """
    dt = tnp1-tn
    if order is None:
      order = self.N
    factor_opts = np.zeros( self.ndata )


    errors = self.eval_error( other_pred=other_pred, tn=tn, tnp1=tnp1, order=order )
    errors = np.maximum( 1e-15, errors )

    for i in range(self.ndata):
        factor_opts[i] = (1./errors[i])**(1/order) #(1./(1.+order))
        # factor by which the time step can be increased while still satisfying the error constraints
    return dt * np.min( factor_opts )

  def suggest_order_error_based(self, other_pred, tn, tnp1):
    # The order of the predictor is adjusted such that the allowed time step is largest for a given relative error level
    # TODO: merge self with eval_timestep to suggest both orders and time step for the next step
    # self strategy is inspired by the strategies described by Petzold, Shampine and others for multistep methods, however
    # here the framework is slightly different: the predictor does not solve and ODE, and here the new order will be selected
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
        factor_opts[:,iord] = self.eval_optimal_timestep(other_pred=other_pred,
                                                         tn=tn,
                                                         tnp1=tnp1,
                                                         order=order)

        # only for debug
        errors[:,iord] = self.eval_error(other_pred=other_pred, tn=tn, tnp1=tnp1, order=order)

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

  def compute_SDN(self, order, with_dt=True):
    """ Compute scaled derivative norm (see Peztold, page 151, which differs from
        Kraft & Schweizer, MSD 2022, eq (30) ) """
    if self.needRefresh:
      self._updateCoef()
    # if self.N<2:
      # return 0
    dt = self._x[0] - self._x[1]
    coeff = self.coef
    assert order<=self.Nacquired
    npoints = order

    nChoice = 0
    if nChoice==0: #approximate formula based on Newton coefficients (misses higher-order contributions)
      res = np.math.factorial(order) * coeff[:,order]

    elif nChoice==1: # true derivatives of the fitting polynomials including all points used by the current active order
      res = np.zeros((self.ndata,))
      for j in range(self.ndata):
        poly = np.polyfit( x=self.x[:self.N],
                               y=self.y[j,:self.N],
                               deg=self.N-1)
        polyder = np.polyder(poly, m=order)
        res[j] = np.polyval(polyder, self.x[0])

    elif nChoice==2: # true derivatives of the smaller fitting polynomials
      res = np.zeros((self.ndata,))
      for j in range(self.ndata):
        poly = np.polyfit( x=self.x[:order+1],
                           y=self.y[j,:order+1],
                           deg=order)
        polyder = np.polyder(poly, m=order)
        res[j] = np.polyval(polyder, self.x[0])

    if with_dt:
      res *= dt**(order-1)
    # res = (dt**order) * np.math.factorial(order) * coeff[:,order-1]
    return abs(res)

  def isPolynomialWellBehaved(self, order):
    sdns = []
    for k in range(1,order): # we do not need the 0-th order term, which is just the last value
      sdns.append( self.compute_SDN(order=k) )
    sdns = np.array(sdns)
    # print(sdns)
    # return np.all( np.diff(sdns.T) < 1e-10 )
    if np.any(sdns[1:-1]==0):
      return False # only the last SDN can be negative
    if sdns[-1]==0.:
      return np.all( sdns[:-2]/sdns[1:-1] > 2. )
    else:
      return np.all( sdns[:-1]/sdns[1:] > 2. )

  def suggest_order_behaviour_based(self):
    if self.nsuccess_with_current_order < self.N:
      if not self.isPolynomialWellBehaved(order=self.N):
        return self.N-1
      else:
        return self.N

    elif self.isPolynomialWellBehaved(order=self.N):
      if self.N < self.NMAX: # TODO: we need to be sure we have one more sampling point
        if self.N < self.Nacquired:
          if self.isPolynomialWellBehaved(order=self.N+1):
            return self.N+1
          else:
            return self.N
        else:
          return self.N+1 # we can't know if the higher order is better,so we just try it !
      else:
        return self.N # we stay at maximum order

    else:
      return self.N-1


  def suggest_order_error_constants_based(self):
    error_constants = np.array([1., 1/2, 5/12, 3/8, 251/720, 95/288, 19087/60480]) # table in Appendix of Meyere, Kraft & Schwiezer, JCND 2021

if __name__=='__main__':
    if 0:
        #% test SDN
        predictor1 = predicteur(NMAX=8)
        predictor1.setTolerances(1e-6, 1e-6)
        # t_test = np.linspace(-1.31, 1.31, predictor1.NMAX+5)
        t_test = np.linspace(0, 1.31, predictor1.NMAX+5)
        poly = [1,0,1,0,1,0,0,0]
        # fun = lambda x: x**5
        fun = lambda x: np.polyval(poly,x)

        for t in t_test:
          predictor1.appendData( x=t, y=fun(t) )

        for i in range(predictor1.N+1):
          print('{0}!f^({0}) = {1} (should be {2})'.format(i,
                                                           predictor1.compute_SDN(order=i, with_dt=False)[0],
                                                           np.polyval( np.polyder(poly, m=i), t_test[-1] ) ) )

        t_plot = np.linspace(t_test[0],t_test[-1],1000)
        plt.figure()
        for i in range(predictor1.N+1):
          yplot = newton_interp_coef(xi=predictor1._x[:i+1], x=t_plot, coef=predictor1._coef[0,:i+1])
          # yplot = newton_interp(xi=predictor1._x[:i+1], yi=predictor1._y[0,:i+1], x=t_plot)
          plt.plot(t_plot, yplot, label='i={}'.format(i))
        plt.plot(t_plot, predictor1.evaluate(t_plot)[0,:], label='pred'.format(i))
        # plt.plot(t_plot, fun(t_plot), label='ref', color=[0,0,0])
        plt.plot(t_test, fun(t_test), label=None, marker='o', linestyle='', color=[0,0,0])
        plt.legend()
        plt.grid()
        plt.ylim(-2,4)

        #%% Test order control
        atol = rtol = 1e-8
        order_max = 10


        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)

        if 0:
          t_plot = np.linspace(0,np.pi,1000)
          fun = lambda t: np.cos(t)
          t_test_global = np.linspace(0,np.pi, 10)
        else:
          t_plot = np.linspace(-4,4,1000)
          fun = lambda t: np.tanh(t)
          dt = 1e-2
          t_test_global = np.flip( np.arange(0, -100*dt, -dt))

        for order in range(1,order_max):
          # t_test = np.linspace(0,np.pi/3,order+3)
          t_test = t_test_global[-order-3:] # same stencil for all cases
          predictor1 = predicteur(NMAX=order)
          predictor1.setTolerances(atol, rtol)
          for t in t_test:
            predictor1.appendData( x=t, y=fun(t) )
          ax1.plot(t_plot, predictor1.evaluate(t_plot)[0,:], label=f'order={order}')

          sdns = np.zeros(order_max-1)
          for k in range(1,order):
            sdns[k] = predictor1.compute_SDN(k)
          print('order=', order, ', SDN=', sdns)
          print('       well behaved at order   :', predictor1.isPolynomialWellBehaved(order=order) )
          print('       well behaved at order+1 :', predictor1.isPolynomialWellBehaved(order=order+1) )
          ax2.semilogy(range(1,order_max), abs(sdns), label=f'order={order}')

        fplot = fun(t_plot)
        ax1.plot(t_plot, fplot, color=[0,0,0], linestyle='--', label='ref')
        ax1.set_ylim(min(fplot)-0.2, max(fplot)+0.2)

        ax1.legend()
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Polynomial fit')
        ax1.grid()

        ax2.legend()
        ax2.set_xlabel('k')
        ax2.set_ylabel('SDN(k)')
        ax2.grid()


    #%%
    if 0:
        #%%
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
            pred.appendData(x=xi[i], y=yi[i], bPerformChecks=False)

          y3 = pred.evaluate(x)[0,:]

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
          predictor.appendData( x=current_x, y=current_y )
          # else:
          #   predictor.replaceLastPoint_single( x=current_x, y=current_y )

          # evaluate on a small interval near the current point
          # xpred = np.linspace(np.min(predictor.x), xtest[i]+3*dx, 100)
          xpred = np.linspace(xtest[0], xtest[-1], 100)
          ypred = predictor.evaluate( x=xpred )[0,:]

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



    #%%test error estimates
    if 0:
        atol = rtol = 1e-8
        t_test = np.array([0, 0.1, 2, 4, 8])
        predictor1 = predicteur(NMAX=t_test.size)
        predictor2 = predicteur(NMAX=t_test.size)
        predictor1.setTolerances(atol, rtol)
        predictor2.setTolerances(atol, rtol)
        for t in t_test:
          predictor1.appendData( x=t, y=t ) #np.cos(t) )
          predictor2.appendData( x=t, y=0. )

        for errmeth in range(3):
          predictor1.error_method = errmeth
          predictor2.error_method = errmeth
          err1 = predictor1.eval_error(other_pred=predictor2, tn=t_test[0], tnp1=t_test[1])
          err2 = predictor2.eval_error(other_pred=predictor1, tn=t_test[0], tnp1=t_test[1])
          print(f'method {errmeth}: err1={err1[0]:2e}, err2={err2[0]:.2e}')


        plt.figure()
        t_eval = np.linspace(t_test[0],t_test[1],100)
        plt.plot(t_eval, predictor1.evaluate(t_eval).T, label='pred1')
        plt.plot(t_eval, predictor2.evaluate(t_eval).T, label='pred2')
        plt.legend()
        plt.grid()
        plt.xlabel('t')
        plt.ylabel('y')

    #%%test error estimates
    if 1:
        atol = rtol = 1e-8
        t_test = np.array([0, 0.1, 0.8, 1.6])
        funtest = lambda x: np.cos(x)

        predictor1 = predicteur(NMAX=t_test.size)
        predictor1.setTolerances(atol, rtol)
        predictor1.error_method = 0

        for t in t_test:
          predictor1.appendData( x=t, y=funtest(t) )

        class dummypred:
          def __init__(self):
            pass
          def evaluate(self,t):
            return funtest(t)

        err = predictor1.eval_error(other_pred=dummypred(), tn=t_test[0], tnp1=t_test[1])
        print(f'err={err[0]:.2e}')

        plt.figure()
        t_eval = np.linspace(t_test[0],t_test[-1],100)
        plt.plot(t_eval, predictor1.evaluate(t_eval).T, label='pred1')
        plt.plot(t_eval, funtest(t_eval), label='input')
        plt.legend()
        plt.grid()
        plt.xlabel('t')
        plt.ylabel('y')