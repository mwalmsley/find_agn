import sys
import os
import json

import numpy as np
import pandas as pd

from astropy.cosmology import WMAP9 as cosmo

import notebook_utils

sys.path.insert(0,'/data/repos/find_agn/easyGalaxy')  # cloned locally because pypi not updated
import ezgal  # linting error - does work

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

    default_model = ezgal.model(model_loc)

    model =  default_model.make_csp(star_formation)  # apply custom star formation history

    # This MUST match the order of notebook_utils.all_bands. Different names, so hard to check.
    # Filters must be added after make_csp because make_csp does not preserve any prev. filters

    for band in ALL_BANDS:
        model.add_filter(band)

    return model


def get_normalised_model_continuum(model, galaxy, observed=False):
    """[summary]
    
    Args:
        model ([type]): [description]
        galaxy ([type]): [description]
        observed (bool, optional): Defaults to False. [description]
    
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


def save_template(model, save_dir, formation_z, current_z, mass=1.):
    """Save galaxy and continuum templates from EZGAL model at defined formation_z, current_z.
    Model SF history MUST match formation_z, current_z. TODO: enforce consistency

    Args:
        model (ezgal.ezgal.ezgal): EZGAL model stellar population of 1 Msun
        save_dir (str): path to directory into which to save templates
        formation_z (float): formation redshift of template. Must match model SF history.
        current_z (float): current redshift of template. Must match model SF history
        mass (float): mass of template to save. Default: 1. Later rescaling of galaxy is trivial.
    """
    galaxy = get_fake_galaxy(model, formation_z, current_z, mass)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    galaxy_loc, sed_loc = get_saved_template_locs(save_dir, formation_z, current_z, mass)

    with open(galaxy_loc, 'w') as f:
        json.dump(galaxy.to_dict(), f)

    sed = get_normalised_model_continuum(model, galaxy)

    # cannot serialise np array, only simple lists
    for key, data in sed.items():
        if isinstance(data, np.ndarray):
            sed[key] = list(data)
    with open(sed_loc, 'w') as f:
        json.dump(sed, f)


def load_saved_template(save_dir, formation_z, current_z, mass):
    """Load saved galaxy and continuum templates from EZGAL model at defined formation_z, current_z.

    Args:
        model (ezgal.ezgal.ezgal): EZGAL model stellar population of 1 Msun
        save_dir (str): path to directory from which to load templates
        formation_z (float): formation redshift of template. Must match model SF history.
        current_z (float): current redshift of template. Must match model SF history
        mass (float): mass of template to save. Default: 1. Later rescaling of galaxy is trivial.

    Return:
        pd.Series: template galaxy in standard format
        dict: of form {frequency, energy at frequency} for template continuum
    """
    galaxy_loc, sed_loc = get_saved_template_locs(save_dir, formation_z, current_z, mass)

    with open(galaxy_loc, 'r') as galaxy_f:
        galaxy = json.load(galaxy_f)

    with open(sed_loc, 'r') as continuum_f:
        continuum = json.load(continuum_f)

    return galaxy, continuum


def get_saved_template_locs(save_dir, formation_z, current_z, mass):
    param_string = 'fz_{:.3}_cz_{:.3}_m_{}'.format(formation_z, current_z, mass)
    galaxy_loc = os.path.join(save_dir, param_string + '_galaxy.txt')
    sed_loc = os.path.join(save_dir, param_string + '_sed.txt')
    return galaxy_loc, sed_loc
