# RobustMeanEstimator
Compute robust mean estimation (in python) using geometric M-estimator in one or multi dimension.

This algorithm have the advantage of being rather fast even in high dimension and almost minimax in heavy-tailed setting.

## Dependencies
python >=3, numpy

optional : matplotlib, neurtu for illustration notebook.
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

    muhat = M_estimator(X, beta=1)
    print(np.abs(muhat))

Or in multi-dimension

    import numpy as np
    from robust_mean import M_estimator

    rng = np.random.RandomState(42)
    X = np.vstack([rng.normal(size=[95,50]),100*np.ones([5,50])])

    muhat = M_estimator(X, beta=1)
    print(np.linalg.norm(muhat))

See the notebook for other examples.

## Reference

Concentration study of M-estimators using the influence function, by Timothée Mathieu, [arxiv:2104.04416](https://arxiv.org/abs/2104.04416)

## License
This package is released under the 3-Clause BSD license.
