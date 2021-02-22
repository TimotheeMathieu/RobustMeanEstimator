cimport cython

import numpy as np
cimport numpy as np
from cython cimport floating
from libc.math cimport sqrt

np.import_array()

cdef floating _euclidean_dense_dense(
        floating* a,  # IN
        floating* b,  # IN
        int n_features):
    """Euclidean distance between a dense and b dense"""
    cdef:
        int i
        int n = n_features // 4
        int rem = n_features % 4
        floating result = 0

    # We manually unroll the loop for better cache optimization.
    for i in range(n):
        result += ((a[0] - b[0]) * (a[0] - b[0])
                  +(a[1] - b[1]) * (a[1] - b[1])
                  +(a[2] - b[2]) * (a[2] - b[2])
                  +(a[3] - b[3]) * (a[3] - b[3]))
        a += 4; b += 4

    for i in range(rem):
        result += (a[i] - b[i]) * (a[i] - b[i])

    return result

ctypedef floating (*f_type)(floating, floating, floating)
cdef floating psihsx(floating x, floating beta, floating p):
    # Function psi(x)/x where psi is Huber score function
    if (x > beta) :
        return beta / x
    elif x < -beta:
        return - beta / x
    else:
        return 1

cdef floating psicsx(floating  x, floating  beta, floating p):
    # Function psi(x)/x where psi is Catoni score function
    if x != 0 :
        return beta*np.log(1+np.abs(x)/beta+x**2/2/beta**2)/np.abs(x)
    else:
        return 1

cdef floating psipsx(floating  x, floating beta, floating  p):
    # Function psi(x)/x where psi is Polynomial score function
    return 1/(1+np.abs(x/beta)**(1-1/p))




@cython.boundscheck(False)  # Deactivate bounds checking
def compute_mu_1D(floating mu, floating[:] X,
                  floating beta, str name, floating p) :

    cdef f_type psisx
    if name == "Huber":
        psisx = psihsx
    elif name == "Catoni":
        psisx = psicsx
    elif name == "Polynomial":
        psisx = psipsx

    cdef int n = len(X)
    cdef floating new_mu = 0
    cdef floating normalizing_const = 0
    cdef int i
    cdef floating wi

    for i in range(n):
        wi = psisx(X[i]-mu, beta, p)
        new_mu +=  wi* X[i]
        normalizing_const += wi

    return new_mu / normalizing_const




def compute_mu_mD( np.ndarray[floating] mu, np.ndarray[floating, ndim=2] X,
                  floating beta, str name, floating p):

    cdef f_type psisx
    if name == "Huber":
      psisx = psihsx
    elif name == "Catoni":
      psisx = psicsx
    elif name == "Polynomial":
      psisx = psipsx

    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef np.ndarray[floating] new_mu = np.zeros(d)
    cdef floating normalizing_const = 0
    cdef int i, j
    cdef floating wi, Ni

    for i in range(n):
        Ni = sqrt(_euclidean_dense_dense(&X[i, 0], &mu[0], d))
        wi = psisx(Ni, beta, p)
        for j in range(d):
            new_mu[j] +=  wi* X[i, j]
        normalizing_const += wi

    return np.array(new_mu) / normalizing_const
