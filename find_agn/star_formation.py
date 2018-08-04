"""Common star formation histories. EZGAL can use these to synthesis the corresponding spectra.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

from astropy.cosmology import WMAP9 as cosmo

from find_agn import ezgal_wrapper


class ExponentialHistory():

    def __init__(self, tau):
        """[summary]
        
        Args:
            tau (float): exponential decay time in Gyrs
        """

        self.tau = tau

    def __call__(self):
        """Get the star formation history at each age for a population with an exponential
        star forming history with decay time tau

        Returns:
            function: get relative star formation rate when called with age (Gyrs)
        """
        return lambda galaxy_age: np.exp(-1.0 * galaxy_age / self.tau)  # expects age in Gyrs


class ConstantBurstHistory():

    def __init__(self, start_age, end_age, relative_peak=1., sigma=1000.):
        self.start_age = start_age
        self.end_age = end_age
        self.relative_peak = relative_peak
        self.sigma = sigma


    def __call__(self):
        """Get function describing star formation at age (Gyrs)
        for a population with a constant star formation
        history between start_age and end_age

        Args:
            start_age (np.array): start time of burst (Gyrs)
            end_age (np.array): end time of burst (Gyrs)
            relative_peak (float, optional): Defaults to 1.. Level of burst SF (EZGAL will normalise)
            sigma (float): width of smoothing filter. High values cause smoother histories.

        Returns:
            function: get relative star formation rate when called with age (Gyrs)
        """
        age_samples = np.linspace(0., 20., 10000)  # EZGAL age always 0-20
        star_formation_samples = np.zeros_like(age_samples)
        sample_at_peak = (age_samples >= self.start_age) & (age_samples <= self.end_age)
        star_formation_samples[sample_at_peak] = self.relative_peak
        smoothed_sf_samples = gaussian_filter1d(star_formation_samples, sigma=self.sigma)
        return interp1d(age_samples, smoothed_sf_samples)


class DualBurstHistory():

    def __init__(self, current_z, formation_z, first_duration, second_duration):
        self.current_z = current_z
        self.formation_z = formation_z
        self.first_duration = first_duration
        self.second_duration = second_duration

        self.age_at_current_z = cosmo.age(self.current_z).value - cosmo.age(self.formation_z).value  # Gyrs
        assert self.age_at_current_z - self.second_duration > self.first_duration  # no overlap allowed

    def __call__(self):
        """Constant starburst at formation and at current z

        Args:
            current_z (float): observed redshift of galaxy
            formation_z (float): redshift at which galaxy formed
            first_duration (float): duration of first starburst in Gyr
            second_duration (float): duration of second starburst in Gyr

        Returns:
            function: get relative star formation rate when called with age (Gyrs)
        """
        first_burst = ConstantBurstHistory(
            start_age=0.,
            end_age=self.first_duration)

        second_burst = ConstantBurstHistory(
            start_age=self.age_at_current_z - self.second_duration,
            end_age=self.age_at_current_z)

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
