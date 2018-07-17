import pytest

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import ezgal_wrapper
import star_formation

@pytest.fixture()
def reference_model_loc():
    return 'test_figures/bc03_ssp_z_0.02_salp.model'

@pytest.fixture()
def formation_z():
    return 2.

@pytest.fixture()
def ref_exponential():
    return lambda x: star_formation.exponential_sfh(x, tau=1.0)


@pytest.fixture()
def ref_burst():
    return lambda x: star_formation.constant_burst_sfh(x, start_age=3., end_age=5.)


@pytest.fixture()
def ref_dual_burst(formation_z):
    return lambda x: star_formation.dual_burst_sfh(
        x,
        current_z=0.05,
        formation_z=formation_z, 
        first_duration=1., 
        second_duration=.5, 
        )


def test_star_formation_profiles(formation_z, ref_exponential, ref_burst, ref_dual_burst):
    galaxy_ages = np.linspace(0., 15., 1000)  # note that universe is not 8 + 15 Gyr old, just to explore
    plt.plot(galaxy_ages, ref_exponential(galaxy_ages), label='Exponential')
    plt.plot(galaxy_ages, ref_burst(galaxy_ages), label='Burst')
    plt.plot(galaxy_ages, ref_dual_burst(galaxy_ages), label='Dual Burst')
    plt.xlabel('Age of Galaxy (Gyr)')
    plt.ylabel('Relative Star Formation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('test_figures/star_formation_profiles.png')


# def test_exponential_sfh_model(reference_model_loc, ref_exponential):
#     model = ezgal_wrapper.get_model(reference_model_loc, ref_exponential)

#     formation_z = 3.0
#     fig, _ = starformation_history.visualise_model_sfh(
#         model,
#         model_name='Tau = 1.0 Gyr',
#         formation_z=formation_z)

#     fig.savefig('test_figures/exponential_sfh_model.png')


# def test_constant_burst_sfh_model(reference_model_loc, ref_burst):
#     model = ezgal_wrapper.get_model(reference_model_loc, ref_burst)

#     formation_z = 8.0
#     fig, _ = starformation_history.visualise_model_sfh(
#         model,
#         model_name='Burst',
#         formation_z=formation_z)

#     fig.savefig('test_figures/constant_burst_sfh_model.png')


def test_dual_burst_sfh_model(reference_model_loc, ref_dual_burst, formation_z):
    model = ezgal_wrapper.get_model(reference_model_loc, ref_dual_burst)

    fig, _ = star_formation.visualise_model_sfh(
        model,
        model_name='Dual Burst',
        formation_z=formation_z)

    fig.savefig('test_figures/dual_burst_sfh_model.png')
