import numpy as np
import pytest
from robust_mean import M_estimator

rng = np.random.RandomState(42)

betas = [1, None]
names = ["Huber", "Catoni", "Polynomial"]
n = 100
epsilon = 0.05
d = 10

# Check good in normal case
@pytest.mark.parametrize("beta", betas)
@pytest.mark.parametrize("name", names)
def test_1D_gauss(beta, name):
    sample = rng.normal(size=n)
    assert np.abs(M_estimator(sample, beta=beta, name=name)) < 4 / np.sqrt(n)


@pytest.mark.parametrize("beta", betas)
@pytest.mark.parametrize("name", names)
def test_multiD_gauss(beta, name):
    sample = rng.normal(size=[100, d])
    assert np.linalg.norm(M_estimator(sample, beta=beta, name=name)) < 4 * np.sqrt(
        10 / d
    )


# Check good in corrupted normal case
@pytest.mark.parametrize("beta", betas)
@pytest.mark.parametrize("name", names)
def test_1D_cor(beta, name):
    sample = np.hstack([rng.normal(size=n), 30 * np.ones(int(n * epsilon))])
    assert (
        np.abs(M_estimator(sample, beta=beta, name=name)) < 4 / np.sqrt(n) + 4 * epsilon
    )


@pytest.mark.parametrize("beta", betas)
@pytest.mark.parametrize("name", names)
def test_multiD_cor(beta, name):
    sample = np.vstack([rng.normal(size=[100, d]), 30 * np.ones([int(n * epsilon), d])])
    assert (
        np.linalg.norm(M_estimator(sample, beta=beta, name=name))
        < 4 * np.sqrt(10 / n) + 4 * np.sqrt(d) * epsilon
    )
