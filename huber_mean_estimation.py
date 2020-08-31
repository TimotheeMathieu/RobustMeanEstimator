import numpy as np
import inspect
from scipy import optimize
import interval
from joblib import Parallel, delayed



class huber:
    """
    Class for Huber estimator in 1D or geometric Huber estimator as described
    in the article in multiD
    Parameters
    ----------
    beta : float or None, default = 1
        Parameter of scale in the estimator. If None, use Lepski's method,
        it can be computationally intensive depending on the value of grid.
        Lepski's method in multi-D is not implemented yet.

    maxiter : int, default = 100
        Maximum number of iterations.

    tol : float, default = 1e-5
        Tolerance for the stopping criterion.

    t : float, default = 6
        Level in Lepski's method. Only used if beta is None.

    grid : int, default = 20
        Number of points in the grid. The grid used is linear from 0 to
        beta_max. beta_max is estimted with Newton's method as described in
        the article. The higher grid is the more time needed to estimate.
    
    n_jobs : int, default = 1
        number of jobs for parallel computing of Lepski's method.

    Returns
    -------
    estimator class object

    Example of utilisation
    ----------------------
    > import numpy as np
    > from huber_mean_estimation import huber
    > rng = np.random.RandomState(42)
    > X = rng.normal(size=100)
    > estimator = huber(beta=None,grid=50,t=1)
    > muhat = estimator.estimate(X)
    > print(np.abs(muhat)) # answer is 0.0016596877308189747
    """

    def __init__(self, beta=1, maxiter=100, tol=1e-5, t=6, grid=20, n_jobs=1):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def estimate(self, X):
        if self.beta is not None:
            estimator = huber_beta_fixed(self.beta, self.maxiter, self.tol)
        else:
            if len(X.shape) == 2:
                estimator = Lepski_multiD(self.t, self.grid, self.n_jobs)
            else:
                assert len(X.shape) == 1
                estimator = Lepski_1D(self.t, self.grid, self.n_jobs)
        return estimator.estimate(X)


class huber_beta_fixed:
    def __init__(self, beta=None, maxiter=100, tol=1e-6):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def psi(self, x, beta):
        if beta != 0:
            return x * (abs(x) < beta) + (abs(x) >= beta) * (2 * (x > 0) - 1) * beta
        else:
            return 2 * (x > 0) - 1

    def psisx(self, x, beta):
        return (abs(x) < beta) + (abs(x) >= beta) * beta / (1e-10 + np.abs(x))

    def estimate(self, X):
        beta = self.beta
        if beta == 0:
            return np.median(X, axis=0)
        # Initialization
        mu = np.median(X, axis=0)
        last_mus = []

        # Iterative Reweighting algorithm
        for f in range(self.maxiter):
            if len(X.shape) == 2:
                # If in multi-dimension, deal with it
                w = self.psisx(np.linalg.norm(X - mu, axis=1), beta)
            else:
                w = self.psisx(np.abs(X - mu), beta)

            mu = np.average(X, axis=0, weights=w/np.sum(w))
            
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
        self.weights_ = w / np.sum(w)

        return mu


def _find_beta_max(psi, tp, x):
    if len(x.shape) == 2:

        def J(b):
            b = max(b, 1)
            est = tp(b)
            return (
                2
                * len(x)
                * np.mean(psi(np.linalg.norm(x - est.estimate(x), axis=1) / b) ** 2)
                - 1
            )

    else:

        def J(b):
            b = max(b, 1)
            est = tp(b)
            return 2 * len(x) * np.mean(psi(np.abs(x - est.estimate(x)) / b) ** 2) - 1

    return optimize.root(J, 1).x[0]


class Lepski_multiD:
    def __init__(self, t=6, grid=None, n_jobs=1):
        raise NotImplementedError


class Lepski_1D:
    def __init__(self, t=6, grid=None, n_jobs=1):
        self.t = t
        if grid is None:
            self.grid = 10
        else:
            self.grid = grid
        self.index_beta = 0
        self.n_jobs = n_jobs

    def estimate(self, X):
        n = len(X)

        def I(beta):
            # Compute the intereval corresponding to beta
            hub = huber_beta_fixed(beta)
            V = np.sum(
                [hub.psi(X[i] - X[j], beta) ** 2 for i in range(n) for j in range(n)]
            ) / (n * (n - 1))
            tp = hub.estimate(X)
            bound = (
                np.sqrt(V * self.t / beta ** 2 / n / 2) + self.t / n / beta
            )  # +np.sqrt(V)/beta
            return (tp - bound, tp + bound)

        beta_min = 0
        hub = huber_beta_fixed()
        beta_max = _find_beta_max(lambda t: hub.psi(t, 1), huber_beta_fixed, X)
        grid = np.linspace(beta_min, beta_max, num=self.grid + 1)[1:]
        intervals = Parallel(n_jobs=self.n_jobs)(delayed(I)( beta)  for beta in grid)
        self.intervals = intervals
        Ichap = self._intersection(intervals)
        self.beta_max = beta_max
        self.betas = grid[: self.index_beta]
        return Ichap.midpoint[0][0]

    def _intersection(self, intervals):
        # Compute the intersection of the intervals
        result = interval.interval[0, np.inf]
        for I in intervals:
            a, b = I
            x = interval.interval[a, b]
            if len(x & result) == 1:
                result = x & result
                self.index_beta += 1
            else:
                break
        return result
