import numpy as np

# Import cython helper.
from ._robust_mean import compute_mu_1D, compute_mu_mD

class M_estimator:
    """Compute geometric M-estimator.

    Parameters
    ----------

    beta : float  default = 1
        scale parameter, must be non-negative.

    name : string in {'Huber', 'Catoni', 'Polynomial'}
        name of the score function used.

    p : float, default = 5
        parameter for Polynomial estimator, must be positive.

    maxiter : int, default = 100
        maximum number of iterations.

    tol : float, default = 1e-5
        tolerance for stopping criterion.

    Examples
    --------
    >>> from robust_mean import M_estimator
    >>> import numpy as np
    >>> mest = M_estimator(1)
    >>> X = np.random.normal(size=[100,100])
    >>> result = mest.estimate(X)

    """
    def __init__(self, beta=1, name='Huber', p=5, maxiter=100, tol=1e-5):
        self.beta = beta
        self.name = name
        self.p = p
        self.maxiter = maxiter
        self.tol = tol


    def estimate(self, x):
        X = np.array(x)
        beta = np.float64(self.beta)

        # Initialization
        dim_1 = (len(X.shape)==1)
        mu = np.median(X, axis=0)
        last_mus = []

        # Checks
        if self.name not in ['Huber', 'Catoni', 'Polynomial']:
            raise ValueError("name must be either Huber, Catoni or Polynomial")
        if self.beta < 0 :
            raise ValueError("beta must be non-negative")
        if self.p < 0 :
            raise ValueError("p must be non-negative")
        if self.tol <= 0 :
            raise ValueError("tol must be positive")


        # Iterative Reweighting algorithm
        for f in range(self.maxiter):
            # Conmpute weights and weighted average step to construct new mu
            if dim_1:
                mu = compute_mu_1D(mu, X, beta, self.name, self.p)
            else:
                mu = compute_mu_mD(mu, X, beta, self.name, self.p)

            # Stopping criterion
            if f>10:
                if np.std(last_mus) < self.tol:
                    break
                else:
                    last_mus.pop(0)
                    last_mus.append(mu)
            else:
                last_mus.append(mu)

        self.beta_ = beta

        return mu
