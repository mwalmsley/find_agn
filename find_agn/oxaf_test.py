import os

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from find_agn import oxaf
from find_agn.tests import TEST_FIGURE_DIR


POSSIBLE_P_NT = [0.1, 0.5, 0.9]  # weighting on non-thermal component
POSSIBLE_GAMMA = np.linspace(1.2, 2.6, 5)  # log space disk slope
POSSIBLE_E_PEAK_EV = np.linspace(18., 100., 5)   # peak disk emission in ev
POSSIBLE_E_PEAK_LOG_KEV = np.log10(POSSIBLE_E_PEAK_EV / 1e3)  # convert from ev to log10(e in kev)


@pytest.fixture(params=POSSIBLE_P_NT)
def p_nt(request):
    return request.param


@pytest.fixture(params=POSSIBLE_GAMMA)
def gamma(request):
    return request.param


@pytest.fixture(params=POSSIBLE_E_PEAK_LOG_KEV)
def e_peak(request):
    """The energy of the peak of the BBB, in log10(E/keV)"""
    return request.param


def test_disk(e_peak):
    bin_centers, bin_widths, flux_in_bins = oxaf.disk(e_peak)


def test_non_thermal(e_peak, gamma):
    bin_centers, bin_widths, flux_in_bins = oxaf.non_thermal(e_peak, gamma)


def test_full_spectrum(e_peak, gamma, p_nt):
    bin_centers, bin_widths, flux_in_bins = oxaf.full_spectrum(e_peak, gamma, p_nt)


def test_oxaf_visual():

    fig, rows = plt.subplots(
        nrows=len(POSSIBLE_P_NT), # p_nt on each row
        ncols=len(POSSIBLE_GAMMA),  # gamma on each column
        figsize=(len(POSSIBLE_GAMMA) * 3, len(POSSIBLE_P_NT) * 2),  # size figure accordingly
        sharex=True,
        sharey=True
        )  

    for row_n, row in enumerate(rows):
        for ax_n, ax in enumerate(row):
            p_nt = POSSIBLE_P_NT[row_n]
            gamma = POSSIBLE_GAMMA[ax_n]
            for e_peak in POSSIBLE_E_PEAK_LOG_KEV:
                bin_centers, bin_widths, flux_in_bins = oxaf.full_spectrum(e_peak, gamma, p_nt)
                ax.loglog(bin_centers, flux_in_bins)  # assumes constant width bins!

    row_labels = ['p_nt={:.3}'.format(x) for x in POSSIBLE_P_NT]
    col_labels = ['gamma={:.3}'.format(x) for x in POSSIBLE_GAMMA]

    for col, col_label in zip(rows[0], col_labels):
        col.set_title(col_label)

    for row, row_label in zip(rows[:, 0], row_labels):
        row.set_ylabel(row_label, size='large')

    for col, col_label in zip(rows[-1], col_labels):
        col.set_xlabel('Approx. bin energy (keV)', size='large')

    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'oxaf_models.png'))
    