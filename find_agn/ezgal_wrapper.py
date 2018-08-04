import sys
import os

import numpy as np
import pandas as pd

from find_agn import notebook_utils

import easyGalaxy.ezgal  # linting error - does work

ALL_BANDS = [
    'galex_fuv',
    'galex_nuv',
    'sloan_u',
    'sloan_g',
    'sloan_r',
    'sloan_i',
    'sloan_z',
    'ukidss_y',
    'ukidss_j',
    'ukidss_h',
    'ukidss_k',
    'wise_ch1',
    'wise_ch2',
    'wise_ch3',
    'wise_ch4'
]

BAND_SUBSET = [
    'galex_fuv',
    'galex_nuv',
    'sloan_g',
    'sloan_r',
    'ukidss_k',
    'wise_ch1',
    'wise_ch4'
] 

def get_fake_galaxy(model, formation_z, current_z, mass):
    """Wrapper to get galaxy magnitudes from an ezgal model and convert to standard table format

    Args:
        model (ezgal.ezgal.ezgal): stellar population model
        formation_z (float): formation redshift of stellar population
        current_z (float): current redshift of stellar population (for age and k-correction)
        mass (float): mass of stellar population (i.e. galaxy) in M_sun

    Returns:
        pd.Series: Galaxy with standard magnitude and flux columns
    """
    magnitudes = get_fake_galaxy_magnitudes(model, formation_z, current_z, mass)
    galaxy = ezgal_magnitudes_to_galaxy(magnitudes)
    galaxy['z'] = current_z
    galaxy['formation_z'] = formation_z
    galaxy['mass'] = mass
    return galaxy

def get_fake_galaxy_magnitudes(model, formation_z, current_z, mass):
    """Scale model stellar population by mass and calculate observed magnitudes

    Args:
        model (ezgal.ezgal.ezgal): stellar population model
        formation_z (float): formation redshift of stellar population
        current_z (float): current redshift of stellar population (for age and k-correction)
        mass (float): mass of stellar population (i.e. galaxy) in M_sun

    Returns:
        np.array: k-corrected absolute mag. by band of stellar pop. at current_z
    """
    # should have no normalisation
    assert model.get_normalization(2.) == 0.0  # 2. is simply a random redshift to check
    mags_for_msun = model.get_observed_absolute_mags(zf=formation_z, zs=current_z, ab=True).squeeze()
    return mags_for_msun - 2.5 * np.log10(mass)  # use 0 index if single current_z input


def ezgal_magnitudes_to_galaxy(magnitudes):
    """Convert magnitudes implicitly ordered by useful_bands to my standard galaxy data format
    TODO It might be sensible to create a Galaxy object instead of using pandas.
    Args:
        magnitudes (np.array): absolute k-corrected magnitudes ordered by notebook_utils.all_bands

    Returns:
        pd.Series: Galaxy with standard magnitude and flux columns
    """

    assert len(magnitudes) == 15

    fake_galaxy = pd.Series(dict(zip(
        [band + '_MAG_ABSOLUTE' for band in notebook_utils.all_bands],
        magnitudes
    )))

    # .value to remove the Jy unit
    for band in notebook_utils.ordered_sdss_bands + notebook_utils.ordered_ukidss_bands:
        fake_galaxy[band + '_FLUX_ABSOLUTE'] = notebook_utils.ab_mag_to_flux(fake_galaxy[band + '_MAG_ABSOLUTE']).value
    for band in notebook_utils.ordered_wise_bands:
        fake_galaxy[band + '_FLUX_ABSOLUTE'] = notebook_utils.ab_mag_to_flux(fake_galaxy[band + '_MAG_ABSOLUTE']).value

    return fake_galaxy




def get_model(model_loc, star_formation):
    """Load an ezgal stellar population model. 
    Add useful filters to measure e.g. magsby default.

    Args:
        model_loc (str): path for stellar population model to load
        star_formation (function): relative star formation given arg of age in Gyr

    Returns:
        ezgal.ezgal.ezgal: model for the evolution over `z` of that stellar population
    """

    default_model = easyGalaxy.ezgal.model(model_loc)

    model =  default_model.make_csp(star_formation)  # apply custom star formation history

    # This MUST match the order of notebook_utils.all_bands. Different names, so hard to check.
    # Filters must be added after make_csp because make_csp does not preserve any prev. filters

    for band in ALL_BANDS:
        model.add_filter(band)

    return model


def get_normalised_model_continuum(model, galaxy, observed=False):
    """[summary]
    
    Args:
        model (ezgal.ezgal.ezgal): ezgal model
        galaxy (dict): with calculated ezgal model, formation_z, z, 
        observed (bool, optional): Defaults to False. If True, return continuum in observed frame.
    
    Returns:
        [type]: [description]
    """
    freq, sed = model.get_sed_z(  # flux density Fv (energy/area/time/photon for each frequency v)
        galaxy['formation_z'],
        galaxy['z'],
        units='Fv',  # in Jy (I think) i.e. 10^-23 erg / s / cm^2 / Hz
        observed=observed,
        return_frequencies=True
        )
    energy_from_mags = notebook_utils.get_spectral_energy(galaxy)['energy']
    energy_from_sed = sed * freq  # energy density v Fv i.e. 10^-23 erg / s / cm^2
    sed_log_offset = np.max(np.log10(energy_from_mags) - np.max(np.log10(energy_from_sed)))

    return {
        'frequency': freq,  # defined at 6900 places. Hz.
        'wavelength': 299792458 / freq,  # lambda (m) = c (ms^1) / freq (Hz)
        'flux_density': sed,  # not shifted in any way
        'energy_density': 10 ** (np.log10(energy_from_sed) + sed_log_offset)  # add a log shift
    }

def bc03_model_name(metallicity=0.05, sfh='ssp', imf='salp'):
    return 'bc03_{}_z_{}_{}.model'.format(sfh, metallicity, imf)