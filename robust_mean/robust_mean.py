import numpy as np

# Import cython helper.
from ._robust_mean import compute_mu_1D, compute_mu_mD
from scipy.spatial.distance import cdist, euclidean


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def M_estimator(X, beta=None, name="Huber", p=5, maxiter=100, tol=1e-5, grid_size=50):
    """Compute geometric M-estimator.

    Parameters
    ----------

    beta : float or None, default = None
        scale parameter, must be non-negative or None.
        If None, the heuristic described in [1] is used (computationally heavy)

    name : string in {'Huber', 'Catoni', 'Polynomial'}, default = "Huber"
        name of the score function used.

    p : float, default = 5
        parameter for Polynomial estimator, must be positive.

    maxiter : int, default = 100
        maximum number of iterations.

    tol : float, default = 1e-5
        tolerance for stopping criterion.

    grid_size : int, default = 50
        Size of the grid on the values of beta used for the heuristic
        choice of beta is beta is None.

    Examples
    --------
    >>> from robust_mean import M_estimator
    >>> import numpy as np
    >>> X = np.random.normal(size=[100,100])
    >>> result = M_estimator(X)

    Reference
    ---------
    [1] "Concentration study of M-estimators using the influence function"
        by Timothée Mathieu
    """
    one_dim = len(np.shape(X)) == 1

    if beta is None:
        if one_dim:
            return adaptive_M(
                X[:, np.newaxis], name, p, grid_size, maxiter, tol
            ).ravel()
        else:
            return adaptive_M(X, name, p, grid_size, maxiter, tol)
    else:
        mest = M_estimator_fixed_param(beta, name, p, maxiter, tol)
        if one_dim:
            return mest.estimate(X[:, np.newaxis]).ravel()
        else:
            return mest.estimate(X)


class M_estimator_fixed_param:
    """Compute geometric M-estimator.

    Parameters
    ----------

    beta : float, default = 1
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

    def __init__(self, beta=1, name="Huber", p=5, maxiter=100, tol=1e-5):
        self.beta = beta
        self.name = name
        self.p = p
        self.maxiter = maxiter
        self.tol = tol

    def estimate(self, x):
        X = np.array(x)
        beta = np.float64(self.beta)

        # Initialization
        dim_1 = len(X.shape) == 1
        mu = np.median(X, axis=0)
        last_mus = []

        # Checks
        if self.name not in ["Huber", "Catoni", "Polynomial"]:
            raise ValueError("name must be either Huber, Catoni or Polynomial")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.p < 0:
            raise ValueError("p must be non-negative")
        if self.tol <= 0:
            raise ValueError("tol must be positive")

        # Iterative Reweighting algorithm
        for f in range(self.maxiter):
            # Conmpute weights and weighted average step to construct new mu
            if dim_1:
                mu = compute_mu_1D(mu, X, beta, self.name, self.p)
            else:
                mu = compute_mu_mD(mu, X, beta, self.name, self.p)

            # Stopping criterion
            if f > 10:
                if np.linalg.norm(np.std(last_mus, axis=0)) < self.tol:
                    break
                else:
                    last_mus.pop(0)
                    last_mus.append(mu)
            else:
                last_mus.append(mu)
        if f == self.maxiter - 1:
            print("Warning, all the iteration have been used")

        self.beta_ = beta

        return mu


def psi(x, name, beta, p):
    if name == "Huber":
        return x * (x <= beta) + beta * (x > beta)
    elif name == "Catoni":
        return beta * np.log(1 + np.abs(x) / beta + x ** 2 / 2 / beta ** 2)
    else:
        return x / (1 + np.abs(x / beta) ** (1 - 1 / p))


def adaptive_M(X, name="Huber", p=1, grid_size=50, maxiter=200, tol=1e-5):
    n = len(X)

    MAD = np.median(np.linalg.norm(X - geometric_median(X), axis=1))

    params = np.linspace(1e-5, MAD * np.sqrt(n), num=grid_size)
    res = []

    if name == "Huber":
        mult_cst = 1
    elif name == "Catoni":
        mult_cst = 5 / (4 * 8)
    else:
        mult_cst = 1 / 16

    def vpsi(X):
        est = M_estimator_fixed_param(beta, name, p, maxiter=maxiter, tol=tol)
        return np.mean(
            psi(np.linalg.norm(X - est.estimate(X), axis=1), name, beta, p) ** 2
        )

    for i, beta in enumerate(params):
        V = vpsi(X)
        res += [2 * V / len(X) + mult_cst * MAD ** 4 / beta ** 2 + (0.05 * beta) ** 2]
    beta = params[np.argmin(res)]
    est = M_estimator_fixed_param(beta, name, p, maxiter=maxiter, tol=tol)
    return est.estimate(X)
