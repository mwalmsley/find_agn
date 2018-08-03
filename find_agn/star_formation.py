"""Common star formation histories. EZGAL can use these to synthesis the corresponding spectra.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

from astropy.cosmology import WMAP9 as cosmo
import ezgal_wrapper


def exponential_sfh(tau):
    """Get the star formation history at each age for a population with an exponential
    star forming history with decay time tau

    Args:
        tau (float): exponential decay time in Gyrs

    Returns:
        function: get relative star formation rate when called with age (Gyrs)
    """
    return lambda galaxy_age: np.exp(-1.0 * galaxy_age / tau)  # expects age in Gyrs


def constant_burst_sfh(start_age, end_age, relative_peak=1.):
    """Get function describing star formation at age (Gyrs)
    for a population with a constant star formation
    history between start_age and end_age

    Args:
        start_age (np.array): start time of burst (Gyrs)
        end_age (np.array): end time of burst (Gyrs)
        relative_peak (float, optional): Defaults to 1.. Level of burst SF (EZGAL will normalise)

    Returns:
        function: get relative star formation rate when called with age (Gyrs)
    """
    age_samples = np.linspace(0., 20., 10000)  # EZGAL age always 0-20
    star_formation_samples = np.zeros_like(age_samples)
    star_formation_samples[(age_samples >= start_age) & (age_samples <= end_age)] = relative_peak
    smoothed_sf_samples = gaussian_filter1d(star_formation_samples, sigma=50.)
    return interp1d(age_samples, smoothed_sf_samples)


def dual_burst_sfh(current_z, formation_z, first_duration, second_duration):
    """Constant starburst at formation and at current z

    Args:
        current_z (float): observed redshift of galaxy
        formation_z (float): redshift at which galaxy formed
        first_duration (float): duration of first starburst in Gyr
        second_duration (float): duration of second starburst in Gyr

    Returns:
        function: get relative star formation rate when called with age (Gyrs)
    """
    age_at_current_redshift = cosmo.age(current_z).value - cosmo.age(formation_z).value  # Gyrs
    assert age_at_current_redshift - second_duration > first_duration  # no overlap allowed

    first_burst = constant_burst_sfh(
        start_age=0.,
        end_age=first_duration)

    second_burst = constant_burst_sfh(
        start_age=age_at_current_redshift - second_duration,
        end_age=age_at_current_redshift)

    return lambda galaxy_age: first_burst(galaxy_age) + second_burst(galaxy_age)


def visualise_model_sfh(model, model_name, formation_z):
    """Display star formation history of model.
    Must have been called with custom star formation (model.make_csp), else model.sfh will be empty.
    TODO add more (all?) filters to plot, perhaps as a param
    
    Args:
        model (ezgal.ezgal.ezgal): ezgal model with custom star formation history
        model_name (str): name of model, for plot legend only
        formation_z (float): formation redshift of model, for age calculations
    
    Returns:
        matplotlib.figure: matplotlib figure
        tuple: of form (magnitude axis, star formation axis)
    """
    fig, (ax0, ax1) = plt.subplots(ncols=2)

    sample_zs = model.get_zs(formation_z)  # sample z's evenly spaced from 0 to formation_z
    age_at_each_z = model.get_age(formation_z, sample_zs)  # times between redshifts
    filters = ezgal_wrapper.BAND_SUBSET
    absolute_mags_by_age = model.get_absolute_mags(formation_z, zs=sample_zs, filters=filters)
    for band_n, band in enumerate(filters):  # plot the abs. magnitudes in that band
        ax0.plot(age_at_each_z, absolute_mags_by_age[:, band_n], label=band)
    ax0.set_xlabel('Galaxy Age (Gyrs)')
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
