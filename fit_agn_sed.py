"""Fit AGN, host and composite models to a spectral energy distribution (freq * flux at freq)
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

import agn_models


def fit_agn_model_to_spectral_data(log_freq, log_energy, visualise=True):
    """Find the best fit parameters for double skew AGN model to a spectral energy distribution 
    
    Args:
        log_freq (np.array): log10 frequencies of SED
        log_energy ([type]): log10 energies (freq * flux_at_freq) of SED
        visualise (bool, optional): Defaults to True. Plot best fitting model.
    
    Returns:
        [np.array]: Best fit parameters
    """


    initial_guess = [.3, 1., 5] + [.3, 1., 5.]
    index_of_param = {
        'gaussian_scale_a': 0,
        'gaussian_skew_a': 1,
        'shift_for_gaussian_a': 2,
        'gaussian_scale_b': 3,
        'gaussian_skew_b': 4,
        'shift_for_gaussian_b': 5,
    }

    fitfunc = lambda p, x: np.log10(double_skew_wrapper(p, x, index_of_param))

    errfunc = lambda p, x, y: fitfunc(p, x) - y  # y distance in log space
    best_fit, _ = scipy.optimize.leastsq(errfunc, initial_guess, args=(log_freq, log_energy))

    for param, index in index_of_param.items():
        print('{}: {}'.format(param, best_fit[index]))

    if visualise:
        freq_continuum = np.linspace(log_freq.min(), log_freq.max(), 100)
        plt.scatter(log_freq, log_energy, marker='+')
        plt.plot(freq_continuum, fitfunc(best_fit, freq_continuum), "r-")
        plt.show()

    return best_fit


def fit_host_model_to_spectral_data(log_freq, log_energy, visualise=True):
    """Find the best fit parameters for host model to a spectral energy distribution 
    
    Args:
        log_freq (np.array): log10 frequencies of SED
        log_energy ([type]): log10 energies (freq * flux_at_freq) of SED
        visualise (bool, optional): Defaults to True. Plot best fitting model.
    
    Returns:
        [np.array]: Best fit parameters
    """

    initial_guess = [15]
    index_of_param = {
        'shift_for_host': 0,
    }

    fitfunc = lambda p, x: np.log10(host_wrapper(p, x, index_of_param))

    errfunc = lambda p, x, y: fitfunc(p, x) - y  # y distance in log space
    best_fit, _ = scipy.optimize.leastsq(errfunc, initial_guess, args=(log_freq, log_energy))

    # freq_continuum is still in real space, just with log-shaped intervals 
    freq_continuum = np.logspace(np.log10(log_freq.min()), np.log10(log_freq.max()), 100)

    for param, index in index_of_param.items():
        print('{}: {}'.format(param, best_fit[index]))

    if visualise:
        plt.scatter(log_freq, log_energy, marker='+')
        plt.plot(freq_continuum, fitfunc(best_fit, freq_continuum))
        plt.legend(['Data', 'Best Host Model'])
        plt.show()

    return best_fit


def fit_composite_model_to_spectral_data(log_freq, log_energy, visualise=True):
    """Find the best fit parameters for composite host + AGN model to a spectral energy distribution 
    
    Args:
        log_freq (np.array): log10 frequencies of SED
        log_energy ([type]): log10 energies (freq * flux_at_freq) of SED
        visualise (bool, optional): Defaults to True. Plot best fitting model.
    
    Returns:
        [np.array]: Best fit parameters
    """

    double_skew_guess = [.3, 1., 5] + [.3, 1., 5.]
    host_guess = [-9.]
    initial_guess = double_skew_guess + host_guess

    index_of_param = {
        'gaussian_scale_a': 0,
        'gaussian_skew_a': 1,
        'shift_for_gaussian_a': 2,
        'gaussian_scale_b': 3,
        'gaussian_skew_b': 4,
        'shift_for_gaussian_b': 5,
        'shift_for_host': 6
    }

    fitfunc = lambda p, x: np.log10(double_skew_wrapper(p, x, index_of_param) + host_wrapper(p, x, index_of_param))

    errfunc = lambda p, x, y: fitfunc(p, x) - y  # y distance in log space
    best_fit, success = scipy.optimize.leastsq(errfunc, initial_guess, args=(log_freq, log_energy))

    log_freq_continuum = np.linspace(log_freq.min(), log_freq.max(), 100)

    for param, index in index_of_param.items():
        print('{}: {}'.format(param, best_fit[index]))

    if visualise:
        plt.plot(log_freq_continuum, fitfunc(best_fit, log_freq_continuum))
        plt.plot(log_freq_continuum, np.log10(host_wrapper(best_fit, log_freq_continuum, index_of_param)))
        plt.plot(log_freq_continuum, np.log10(double_skew_wrapper(best_fit, log_freq_continuum, index_of_param)))
        plt.scatter(log_freq, log_energy, marker='+')
        plt.legend(['Combined Model', 'Host Model', 'AGN Model', 'Data'])
        plt.show()

    return best_fit, success


def double_skew_wrapper(p, log_freq, mapping):
    """Get (linear) energy expected from double skew AGN model evaluated at log_freq,
    with model parameters p.
    Many optimisers require a list of paramaters. Mapping is used to interpret each parameter.

    Args:
        p (np.array): all model parameters used by optimiser (may include other models)
        log_freq (np.array): log10 of frequency to evaluate model at
        mapping (dict): index in p of each parameter e.g. {'gaussian_scale_a': 0, ...}

    Returns:
        np.array: (linear) energy expected from double skew AGN model evaluated at log_freq
    """
    torus = agn_models.skew_normal(
        log_freq,
        loc=14.,
        scale=p[mapping['gaussian_scale_a']],
        skew=p[mapping['gaussian_skew_a']])
    shifted_torus = 10 ** (np.log10(torus) + p[mapping['shift_for_gaussian_a']])

    disk = agn_models.skew_normal(
        log_freq,
        loc=14.5,
        scale=p[mapping['gaussian_scale_b']],
        skew=p[mapping['gaussian_skew_b']])
    shifted_disk = 10 ** (np.log10(disk) + p[mapping['shift_for_gaussian_b']])

    return  shifted_torus + shifted_disk


def host_wrapper(p, x, mapping):
    """Get (linear) energy expected from host galaxy model evaluated at log_freq,
    with model parameters p.
    Many optimisers require a list of paramaters. Mapping is used to interpret each parameter.

    Args:
        p (np.array): all model parameters used by optimiser (may include other models)
        log_freq (np.array): log10 of frequency at which to evaluate model
        mapping (dict): index in p of each parameter e.g. {'shift_for_host': 0, ...}

    Returns:
        np.array: (linear) energy expected from host galaxy model model evaluated at log_freq
    """
    return 10 ** (np.log10(agn_models.host_flux(x)) + p[mapping['shift_for_host']])


# TODO: make func for taking a dict of params, and creating 
# 1. array of params
# 2. dict of keys and values of indices of params in array