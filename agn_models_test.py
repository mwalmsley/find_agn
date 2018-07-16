"""Visual checks of AGN models
"""


import pytest

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import agn_models
import notebook_utils

@pytest.fixture()
def seyfert():
    return pd.Series(
        {
            'F_FLUX_ABSOLUTE': 6779745300.0,
            'N_FLUX_ABSOLUTE': 29586205000.0,
            'U_FLUX_ABSOLUTE': 184516000000.0,
            'G_FLUX_ABSOLUTE': 605733800000.0,
            'R_FLUX_ABSOLUTE': 1139844400000.0,
            'I_FLUX_ABSOLUTE': 1541716000000.0,
            'Z_FLUX_ABSOLUTE': 1967715900000.0,
            'Y_FLUX_ABSOLUTE': 1541597488519.5886,
            'J_FLUX_ABSOLUTE': 2756578970722.3555,
            'H_FLUX_ABSOLUTE': 4965584929349.648,
            'K_FLUX_ABSOLUTE': 8068094319405.562,
            'W1_FLUX_ABSOLUTE': 2711896687637.9365,
            'W2_FLUX_ABSOLUTE': 1892982051133.9302,
            'W3_FLUX_ABSOLUTE': 5889237230409.728,
            'W4_FLUX_ABSOLUTE': 11202585177323.912
        }
    )


# @pytest.fixture()
# def starforming():
#     return pd.Series(
#         {
#             'F_FLUX_ABSOLUTE': 2004529000.0,
#             'N_FLUX_ABSOLUTE': 1813295400.0,
#             'U_FLUX_ABSOLUTE': 33691814000.0,
#             'G_FLUX_ABSOLUTE': 84510610000.0,
#             'R_FLUX_ABSOLUTE': 144881060000.0,
#             'I_FLUX_ABSOLUTE': 201967520000.0,
#             'Z_FLUX_ABSOLUTE': 225130100000.0,
#             'Y_FLUX_ABSOLUTE': 103771341518.61148,
#             'J_FLUX_ABSOLUTE': 169230051303.45352,
#             'H_FLUX_ABSOLUTE': 307948502445.84216,
#             'K_FLUX_ABSOLUTE': 405581382265.4065,
#             'W1_FLUX_ABSOLUTE': 257383093951.31976,
#             'W2_FLUX_ABSOLUTE': 155759118043.81952,
#             'W3_FLUX_ABSOLUTE': 876932273541.0438,
#             'W4_FLUX_ABSOLUTE': 2164837270796.9597
#         }
#     )


@pytest.fixture()
def reference_galaxy(seyfert):
    sed = notebook_utils.get_spectral_energy(seyfert)
    freq = np.log10(sed['frequency'])
    energy = np.log10(sed['energy'])
    return freq, energy

def test_skew_normal_model(reference_galaxy):
    log_freq = np.linspace(13, 14.5)
    plt.plot(log_freq, np.log10(agn_models.skew_normal(log_freq, 14., .3, 1.)))

    log_ref_freq, log_ref_energy = reference_galaxy
    plt.plot(log_ref_freq, log_ref_energy - 25, 'ro')
    plt.savefig('test_figures/test_torus_model.png')
    # we expect the model to have a recognisable shape vs. the galaxy, at a significant y offset
    plt.clf() 

def test_host_model(reference_galaxy):
    x = np.logspace(13.5, 15., 100)  # logspace is oddly spaced linear space of 10 ** edges
    log_x = np.log10(x)
    plt.plot(log_x, np.log10(agn_models.host_flux(log_x)))
    # we expect the model to have a recognisable shape vs. the galaxy, at a significant y offset
    log_ref_freq, log_ref_energy = reference_galaxy
    plt.plot(log_ref_freq, log_ref_energy - 10, 'ro')
    plt.savefig('test_figures/test_host_model.png')
