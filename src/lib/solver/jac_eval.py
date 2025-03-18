#%% Create a method for computing the Jacobian in an optimised manner, exploiting its sparsity pattern
import numpy as np
import scipy.sparse
import scipy.optimize._numdiff

from src.lib.model import _1d_lib_dfv_model as model
from src.lib.reader import  reader as params

uband=6; lband=-uband
offsets = [i for i in range(lband,uband)]
sparsity_pattern = scipy.sparse.diags(diagonals=[np.ones((3*params.n_all - abs(i))) for i in offsets], offsets=offsets) 

jacfun = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                    fun=model.func_U,
                    x0=x, method='2-point', sparsity=sparsity_pattern,
                    rel_step=1e-8, abs_step=1e-8)



'''
if 0: # test correctness of the sparse Jacobian against a naive dense estimation
  jacfun_full = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                      fun=model.func_U,
                      x0=x, method='2-point', sparsity=None,
                      rel_step=1e-8, abs_step=1e-8)
  
  xtest = X0 + np.random.rand(X0.size).reshape(X0.shape)*(1e-3 + 1e-3*X0)
  
  jac_full = jacfun_full(0.,xtest)
  jac_sparse = jacfun(0.,xtest)
  assert np.max(np.abs(jac_sparse-jac_full)) < 1e-12, 'The sparse Jacobian estimation is not correct'

#%%
'''
'''
if 0:
    #%% JACOBIAN ANALYSIS (sparsity pattern, eigenvalues)
    tempjac = jacfun
    # tempfun = lambda x: getVarsFromX(x=x, options=options)[0] # rho
    # tempjac = lambda t,x: scipy.optimize._numdiff.approx_derivative(
    #                 fun=tempfun,
    #                 x0=x, method='2-point', sparsity=None,
    #                 rel_step=1e-8)
    Xtest = X0 + np.random.rand(X0.size).reshape(X0.shape)*(1e-3 + 1e-3*X0)

    Jac = np.array(tempjac(0., Xtest))
    # Jac = jacfun(out.t[-1], out.y[:,-1])
    plt.figure()
    plt.spy(Jac)
    n_rank_jac = np.linalg.matrix_rank(Jac),
    plt.title('Jacobian (rank={}, shape={})'.format(n_rank_jac, np.shape(Jac)))
    plt.show()
    if n_rank_jac[0]!=np.size(Jac,1):
        print('The following rows of the Jacobian are nil:\n\t{}'.format( np.where( (Jac==0).all(axis=1) ) ))
        print('The following columns of the Jacobian are nil:\n\t{}'.format( np.where( (Jac==0).all(axis=0) ) ))
    if np.size(Jac,1)<500:
        try:
            eigvals, eigvecs= np.linalg.eig(Jac)
            plt.figure()
            plt.scatter(np.real(eigvals), np.imag(eigvals))
            plt.title('Eigenvalues')
        except Exception as e:
            print('caught exception "{}" while computing eigenvalues of the Jacobian'.format(e))
    else:
        print('Skipping eigenvalues computation due to matrix size')
    raise Exception('debug jac')
'''