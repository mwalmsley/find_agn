"""
Utilities that are shared between AGN ipynb notebooks
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo
from astropy import units

ordered_sdss_bands = ['F', 'N', 'U', 'G', 'R', 'I', 'Z']
ordered_ukidss_bands = ['Y', 'J', 'H', 'K']
ordered_wise_bands = ['W1', 'W2', 'W3', 'W4']

all_bands = ordered_sdss_bands + ordered_ukidss_bands + ordered_wise_bands


def ab_mag_to_flux(mags):
    """Given magnitudes measured in the AB system, calculate the corresponding
     flux in Janksy

    Args:
        mags (np.array): Magnitudes in AB system

    Returns:
        [np.array]: Flux of those magnitudes in Jansky
    """
    zero_mag_flux_densities = 3316  # AB zero point is 3316 Jy

    # Jy IS in units, linting error
    return zero_mag_flux_densities * np.power(10., -mags/2.5) * units.Jy



def wise_mag_to_flux(mags, band):
    """Given magnitudes measured WISE bands, calculate the corresponding flux
     in Janksy

    From [here](http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#example):
    > The source flux density, in Jansky [Jy] units, is computed from the
    calibrated WISE magnitudes, mvega using:

    $$F_v[Jy] = F_{v0} * 10^{(-m_{vega} / 2.5)}$$

    > where is the zero magnitude flux density corresponding to the constant
    that gives the same response as that of Alpha Lyrae (Vega).

     $m_{vega}$ is the measured WISE magnitude, expressed in the Vega system.

    > For sources with steeply rising MIR spectra or with spectra that deviate
    significantly from Fν=constant, including cool asteroids and dusty
    star-forming galaxies, a color correction is required

    Args:
        mags (np.array): Magnitudes from WISE AllSky data release in filter `band`
        band (str): WISE filter of magnitudes

    Returns:
        np.array: Flux of those magnitudes in Janksy
    """
    zero_mag_flux_densities = {  # should be in Jy
        'W1': 309.540,
        'W2': 171.787,
        'W3': 31.674,
        'W4': 8.363
    }

    return zero_mag_flux_densities[band] * np.power(10., -mags/2.5)



def formation_redshift_for_fixed_age(current_z, current_galaxy_age):
    """Find the formation redshift for a galaxy of a given redshift and current age
    Useful to set the formation redshift for starburst models to correspond to
    a desired galaxy age
    TODO what type is astropy Gyr units?
    Args:
        current_z (float): current redshift of galaxy
        current_galaxy_age (astropy.Quantity): current age of galaxy. Note: Astropy units!
    
    Returns:
        float: formation redshift of galaxy such that it has current age at current redshift
    """
    universe_age_at_current_redshift = cosmo.age(current_z)
    assert universe_age_at_current_redshift > 12 * units.Gyr
    universe_age_at_formation_redshift = universe_age_at_current_redshift - current_galaxy_age
    assert universe_age_at_formation_redshift < 6 * units.Gyr
    formation_z = z_at_value(cosmo.age, universe_age_at_formation_redshift)
    assert formation_z > 1
    return formation_z



def get_spectral_energy(galaxy):
    """Convert galaxy fluxes into spectral energy

    Args:
        galaxy (pd.Series): Galaxy to measure SED. Must include flux columns.

    Returns:
        dict: of form {flux, frequency, energy, band}
    """


    # for GALEX (FN), see http://www.galex.caltech.edu/researcher/techdoc-ch1.html Table 2
    # double check ugriz - from SDSS DR2 and don't agree with UKIDSS paper Hewett
    band_wavelengths = {
        'F': 1528,
        'N': 2271,
        'U': 3543,
        'G': 4770,
        'R': 6231,
        'I': 7625,
        'Z': 9134,
        'Y': 10305,
        'J': 12483,
        'H': 16313,
        'K': 22010,
        'W1': 33680,
        'W2': 46180,
        'W3': 120820,
        'W4': 221940
    }  # in angstroms

    wavelength = np.zeros(len(all_bands))
    flux = np.zeros_like(wavelength)


    for band_n, band in enumerate(all_bands):
        flux[band_n] = galaxy[band + '_FLUX_ABSOLUTE']
        wavelength[band_n] = band_wavelengths[band]* 1e-10 # convert from angstrom to meter

    frequency = 3e8  / wavelength # frequency in Hz
    energy = flux * frequency  # in Jy Hz

    return {
        'flux': flux,
        'frequency': frequency,
        'energy': energy,
        'band': all_bands
        }


def plot_spectral_energy(data, ax=None, add_labels=False):
# data should have flux, frequency, energy and band. See `get_spectral_energy`.
            
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(np.log10(data['frequency']), np.log10(data['energy']), marker='+')
    ax.set_xlabel('Log $\nu$ (Hz)')
    ax.set_ylabel('$ Log \nu F_{/nu}$ (Jy Hz)')
    
    if add_labels:
        add_band_labels(data, ax)


def add_band_labels(data, ax, y=28.5):
    for band_n, band in enumerate(all_bands):
        ax.text(x=np.log10(data['frequency'][band_n]), y=y, s=band)
