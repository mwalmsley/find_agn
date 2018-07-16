import sys

import numpy as np
import pandas as pd

import notebook_utils

sys.path.insert(0,'/data/repos/find_agn/easyGalaxy')  # cloned locally because pypi not updated
import ezgal  # linting error - does work



def get_fake_galaxy(model, formation_z, current_z, mass):
    """Wrapper to get galaxy magnitudes from an ezgal model and convert to standard table format
    
    Args:
        model (ezgal.ezgal.ezgal): stellar population model
        formation_z (float): formation redshift of stellar population
        current_z (float): current redshift of stellar population (for age and k-correction)
        mass (float): mass of stellar population (i.e. galaxy) in M_sun
    
    Returns:
        [type]: [description]
    """

    magnitudes = get_fake_galaxy_magnitudes(model, formation_z, current_z, mass)
    return ezgal_magnitudes_to_galaxy(magnitudes)


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
    mags_for_1_msun_pop = model.get_observed_absolute_mags(zf=formation_z, zs=current_z, ab=True)
    return mags_for_1_msun_pop[0] - 2.5 * np.log10(mass)  # use 0 index if single current_z input


def ezgal_magnitudes_to_galaxy(magnitudes):
    """Convert magnitudes implicitly ordered by useful_bands to my standard galaxy data format
    # TODO It might be sensible to create a Galaxy object instead of using pandas.
    Args:
        magnitudes (np.array): absolute k-corrected magnitudes ordered by notebook_utils.all_bands
    
    Returns:
        [type]: [description]
    """

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




def get_model(model_loc):
    """Load an ezgal stellar population model. Add useful filters to measure e.g. magsby default.

    Args:
        model_loc (str): path for stellar population model to load

    Returns:
        ezgal.ezgal.ezgal: model for the evolution over `z` of that stellar population
    """

    model = ezgal.model(model_loc)

    # This MUST match the order of notebook_utils.all_bands. Different names, so hard to check.
    useful_bands = [
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
    for band in useful_bands:
        model.add_filter(band)

    return model


def get_normalised_model_continuum(model, galaxy):
    # not currently used
    sed = model.get_sed(3., units='Fv')
    energy_from_mags = notebook_utils.get_spectral_energy(galaxy)['energy']

    sed_offset = np.max(np.log10(energy_from_mags) - np.max(np.log10(model.get_sed(3., units='Fv') * model.vs)))
    return {
        'frequency': model.vs,  # defined at 6900 places
        'energy': 10 ** (np.log10(model.get_sed(3., units='Fv') * model.vs) + sed_offset)  # constant
    }
