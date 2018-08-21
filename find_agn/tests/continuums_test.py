import pytest
import numpy as np

from find_agn import continuums


def test_continuum_fit_background(ref_continuum_raw):
    assert ref_continuum_raw.background is None
    ref_continuum_raw.fit_background()
    assert ref_continuum_raw.background.mean() < ref_continuum_raw.flux_density.max()
    assert ref_continuum_raw.background.mean() > ref_continuum_raw.flux_density.min()
    assert np.abs(ref_continuum_raw.flux_density_subtracted).mean() < 100


def test_continuum_get_bpt_lines(ref_continuum_raw):
    bpt_data = ref_continuum_raw.get_bpt_lines()
    # TODO some asserts should be here
    