import pytest
import numpy as np

from find_agn import bpt_utilities

def test_bpt_sample_grid():
    axes, (x_samples, y_samples), pairs = bpt_utilities.bpt_sample_grid(100)
    assert x_samples.shape == (10, 10)
    assert y_samples.shape == (10, 10)
    assert np.allclose(pairs[0], np.array([-1.5, -1.2]))
    assert np.allclose(pairs[-1], np.array([0.5, 1.2]))


def test_all_permutations():
    a = np.linspace(-1., 2.)
    b = np.linspace(1., 2.)
    perms = bpt_utilities.all_permutations(a, b)
    print(perms.shape)