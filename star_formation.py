"""Common star formation histories. EZGAL can use these to synthesis the corresponding spectra.
"""

import pickle
import json

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from astropy.cosmology import WMAP9 as cosmo


def exponential_sfh(t, tau):
    """Get the star formation history at each age for a population with an exponential 
    star forming history with decay time tau

    Args:
        t (np.array): population ages in Gyrs
        tau (float): exponential decay time in Gyrs

    Returns:
        np.array: relative star formation rate at each t value
    """

    return np.exp(-1.0 * t / tau)


def constant_burst_sfh(galaxy_age, start_age, end_age, relative_peak=1.):
    """Get the star formation history at each age for a population with a constant star formation 
    history between start_t and end_t

    Args:
        galaxy_age (np.array): population ages in Gyrs
        start_age (np.array): start time of burst (Gyrs)
        end_age (np.array): end time of burst (Gyrs)
        relative_peak (float, optional): Defaults to 1.. Level of burst SF (EZGAL will normalise)

    Returns:
        np.array: relative star formation rate at each t value
    """

    star_formation = np.zeros_like(galaxy_age)
    star_formation[(galaxy_age >= start_age) & (galaxy_age <= end_age)] = relative_peak
    return star_formation


def dual_burst_sfh(t, current_z, formation_z, first_duration, second_duration):
    """Constant starburst at formation and at current z
    
    Args:
        t (np.array): galaxy age in years, to retrieve star formation at
        current_z (float): observed redshift of galaxy
        formation_z (float): redshift at which galaxy formed
        first_duration (float): duration of first starburst in Gyr
        second_duration (float): duration of second starburst in Gyr
    
    Returns:
        np.array: relative star formation rate at each t value
    """
    age_at_current_redshift = cosmo.age(current_z).value - cosmo.age(formation_z).value  # Gyrs
    assert age_at_current_redshift - second_duration  > first_duration  # no overlap allowed

    first_burst = constant_burst_sfh(
        t,
        start_age=0.,
        end_age=first_duration)

    second_burst = constant_burst_sfh(
        t,
        start_age=age_at_current_redshift - second_duration,
        end_age=age_at_current_redshift)
    return first_burst + second_burst


def visualise_model_sfh(model, model_name, formation_z):

    # TODO add more (all?) filters to plot, perhaps as a param
    
    fig, (ax0, ax1) = plt.subplots(ncols=2)

    sample_zs = model.get_zs(formation_z)  # sample z's evenly spaced from 0 to formation_z
    age_at_each_z = model.get_age(formation_z, sample_zs)  # times between redshifts
    absolute_mags_by_age = model.get_absolute_mags(formation_z, zs=sample_zs, filters='sloan_u')

    ax0.plot(age_at_each_z, absolute_mags_by_age, label=model_name)
    ax0.set_xlabel('Age (Gyrs)')
    ax0.set_ylabel('Absolute Mag (SDSS u)')
    ax0.legend()

    model_ages_gyr = model.ages * 1e-9  # every calculated age of galaxy, 0 at formation. To Gyr. 
    model_sf_history = model.sfh  # sfh at every age in model.ages

    ax1.plot(model_ages_gyr, model_sf_history, 'k--', label=model_name)
    ax1.set_xlabel('Age (Gyrs)')
    ax1.set_ylabel('SFR')
    ax1.legend()

    fig.tight_layout()

    return fig, (ax0, ax1)
