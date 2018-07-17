import numpy as np
import scipy
from scipy.stats import skewnorm

import ezgal_wrapper


def skew_normal(log_freq, loc, scale, skew):
    """Skewed normal distribution in log space, returned in linear space

    Args:
        log_freq ([type]): [description]
        loc ([type]): [description]
        scale ([type]): [description]
        skew ([type]): [description]

    Returns:
        np.array: Skewed normal distribution constructed in log space, returned in linear space
    """
    return 10 ** skewnorm.pdf(log_freq, loc=loc, scale=scale, a=skew)


def host_flux(log_freq_to_eval): 
    """
    Reproduce model 'normal' galaxy energy distribution
    Args:
        freq_to_eval (np.array): frequencies to evaluate energy distribution, in linear space
    
    Returns:
        np.array: energy at given frequencies, in linear space
    """
    # stored template
    energy = np.array([2.03664191e+13, 2.68862243e+13, 3.97209582e+14, 1.54654660e+15,
       2.60958765e+15, 3.09834791e+15, 3.48257669e+15, 3.99489981e+15,
       3.59104826e+15, 3.36935892e+15, 1.92027459e+15, 6.64356422e+14,
       2.90566634e+14, 2.60055731e+13, 3.79667204e+12])

    frequency = np.array([1.96335079e+15, 1.32100396e+15, 8.46740051e+14, 6.28930818e+14,
       4.81463649e+14, 3.93442623e+14, 3.28443179e+14, 2.91120815e+14,
       2.40326845e+14, 1.83902409e+14, 1.36301681e+14, 8.90736342e+13,
       6.49631875e+13, 2.48303261e+13, 1.35171668e+13])

    # interpolate simply to not worry about float eval errors and plotting
    # interpolate in linear space for simplicity (may look odd in log space)
    energy_continuum = interpolate_energy(frequency=frequency, energy=energy)
    energy_continuum_values = energy_continuum(10 ** log_freq_to_eval)
    return energy_continuum_values


def interpolate_energy(frequency, energy):
    """Allow SED measured at distinct bands to be evaluated anywhere - but badly elsewhere

    Purely for func evaluation convenience, not actual interp.
    Should only be done on the fake galaxies

    Args:
        frequency ([type]): [description]
        energy ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert len(energy) < 20  # intended to make coarse-grained eval not crash, not be 'real' interp
    return scipy.interpolate.interp1d(
        frequency,
        energy,
        kind='linear')  # should be obviously bad if evaluated incorrectly
