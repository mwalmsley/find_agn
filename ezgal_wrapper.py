import sys

import numpy as np
import pandas as pd
import scipy

import notebook_utils

sys.path.insert(0,'/data/repos/find_agn/easyGalaxy')  # cloned locally because pypi not updated
import ezgal  # linting error - does work



def get_model(model_loc):
    model = ezgal.model(model_loc)

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


def get_fake_galaxy_magnitudes(model):
    return model.get_absolute_mags( 3.0, zs=[0.], ab=True)


def ezgal_mags_to_galaxy(mags):
    # convert to same format as bpt_df galaxies
    fake_galaxy = pd.Series(dict(zip(
        [band + '_MAG_ABSOLUTE' for band in notebook_utils.all_bands],
        mags[0]
    )))

    # .value to remove the Jy unit
    for band in notebook_utils.ordered_sdss_bands + notebook_utils.ordered_ukidss_bands:
        fake_galaxy[band + '_FLUX_ABSOLUTE'] = notebook_utils.ab_mag_to_flux(fake_galaxy[band + '_MAG_ABSOLUTE']).value
    for band in notebook_utils.ordered_wise_bands:
        fake_galaxy[band + '_FLUX_ABSOLUTE'] = notebook_utils.ab_mag_to_flux(fake_galaxy[band + '_MAG_ABSOLUTE']).value

    return fake_galaxy


def get_fake_galaxy(model):
    magnitudes = get_fake_galaxy_magnitudes(model)
    return ezgal_mags_to_galaxy(magnitudes)


def get_normalised_model_continuum(model, galaxy):
    sed = model.get_sed(3., units='Fv')
    energy_from_mags = notebook_utils.get_spectral_energy(galaxy)['energy']

    sed_offset = np.max(np.log10(energy_from_mags) - np.max(np.log10(model.get_sed(3., units='Fv') * model.vs)))
    return {
        'frequency': model.vs,  # defined at 6900 places
        'energy': 10 ** (np.log10(model.get_sed(3., units='Fv') * model.vs) + sed_offset)  # constant
    }


def interpolate_energy(frequency, energy):  # purely for func evaluation convenience, not actual interp.
    # should only be done on the fake galaxies

    assert len(energy) < 20  # intended to make coarse-grained eval not fail
    return scipy.interpolate.interp1d(
        frequency, 
        energy,
        kind='linear')  # should be obviously bad