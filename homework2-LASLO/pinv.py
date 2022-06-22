from logging import getLogger
import numpy as np
import numpy.linalg as linalg

'''############################'''
'''Moore Penrose Pseudo Inverse'''
'''############################'''

'''
Compute Moore Penrose Pseudo Inverse
Input: X: matrix to invert
       tol: tolerance cut-off to exclude tiny singular values (default=1e15)
Output: Pseudo-inverse of X.
Note: Do not use scipy or numpy pinv method. Implement the function yourself.
      You can of course add an assert to compare the output of scipy.pinv to your implementation
'''
def compute_pinv(X=None,tol=1e-15):

      #compute the singular value decomposition 
      L, D, R = linalg.svd(X, full_matrices = True)

      #accounting for the tolerance 
      D[np.where (D<=tol)] = 0

      #compute reciprocals 
      D_plus = 1/D

      #arrange reciprocals in a matrix 
      S = np.zeros((R.T.shape[1], L.T.shape[0]))
      S [:D_plus.shape[0], :D_plus.shape[0]]= np.diag(D_plus)

      #compute the pseudo-inverse
      X_plus = np.dot(R.T, np.dot(S, L.T))

      return X_plus
