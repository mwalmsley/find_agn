import pytest

import numpy as np
import galaxy_zoo_accuracy

@pytest.fixture()
def mu():
    return np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_p_mu_uniform():
    n_mu = 100
    mu = np.nan  # uniform p_mu does not depend on mu
    p_mu = galaxy_zoo_accuracy.p_mu_uniform(n_mu, mu)
    assert p_mu.max() < 0.5
    assert p_mu.min() > 0
    assert len(p_mu) == n_mu
    assert all(p_mu == p_mu.min())  # should all have the same value
    assert pytest.approx(np.sum(p_mu), 1)

def test_p_bin_over_half(mu):
    result = galaxy_zoo_accuracy.p_bin_over_half(n_volunteers=40, mu=mu)
    assert result.max() < 0.5
    assert (result.min() < 0.001) and (result.min() > -1e6)

    mu = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    p_bin_over_half = list(map(lambda x: galaxy_zoo_accuracy.p_bin_over_half(40, x), mu))
    assert all(np.diff(p_bin_over_half) > 0)  # should be monotonically increasing

    odd_vols = [3, 5, 7, 9, 11]
    p_bin_over_half = list(map(lambda x: galaxy_zoo_accuracy.p_bin_over_half(x, 0.45), odd_vols))
    assert all(np.diff(p_bin_over_half) > 0)  # should be monotonically increasing

    even_vols = [2, 4,6, 8, 10]
    p_bin_over_half = list(map(lambda x: galaxy_zoo_accuracy.p_bin_over_half(x, 0.45), even_vols))
    assert all(np.diff(p_bin_over_half) > 0)  # should be monotonically increasing


def test_false_positive_acc():
    result = galaxy_zoo_accuracy.false_positive_acc(n_volunteers=40, n_mu=100)
    assert result > 0
    assert result < 0.1