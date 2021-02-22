# RobustMeanEstimator
Compute robust mean estimation (in python) using huber estimator in one or multi dimension.

This algorithm have the advantage of being very fast even in high dimension but it is not always theoretically minimax.

## Dependencies 
python >=3, numpy, scipy, joblib, interval

## Usage
To use, download huber_mean_estimation.py in your working directory or clone 
the git repo. 

Install the library with 
    pip install --user .


Or, if you don't want to install, you have to compile the code with

    python setup.py build_ext --inplace


Then, you can compute a robust estimator of the mean. 
In one dimension

    import numpy as np
    from robust_mean import M_estimator

    rng = np.random.RandomState(42)
    X = np.hstack([rng.normal(size=95),100*np.ones(5)])

    estimator = M_estimator(delta=1)
    muhat = estimator.estimate(X)
    print(np.abs(muhat))

Or in multi-dimension

    import numpy as np
    from robust_mean import M_estimator

    rng = np.random.RandomState(42)
    X = np.vstack([rng.normal(size=[95,50]),100*np.ones([5,50])])

    estimator = M_estimator(delta=1)
    muhat = estimator.estimate(X)
    print(np.linalg.norm(muhat))

See the notebook for another example.

## License
This package is released under the 3-Clause BSD license.
