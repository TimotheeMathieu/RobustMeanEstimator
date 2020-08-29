# RobustMeanEstimator
Compute robust mean estimation (in python) using huber estimator in one or multi dimension.


## Dependencies 
python >=3, numpy, scipy, interval

## Usage
To use, download huber_mean_estimation.py in your working directory or clone 
the git repo.

Then, you can compute a robust estimator of the mean. 
In one dimension

    import numpy as np
    from huber_mean_estimation import huber

    rng = np.random.RandomState(42)
    X = np.hstack([rng.normal(size=95),100*np.ones(5)])
    estimator = huber(delta=1)
    muhat = estimator.estimate(X)
    print(np.abs(muhat))

Or in multi-dimension

    import numpy as np
    from huber_mean_estimation import huber

    rng = np.random.RandomState(42)
    X = np.vstack([rng.normal(size=[95,50]),100*np.ones([5,50])])
    estimator = huber(delta=1)
    muhat = estimator.estimate(X)
    print(np.linalg.norm(muhat))
