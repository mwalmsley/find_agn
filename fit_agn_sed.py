import numpy as np
import agn_models
import scipy.optimize
import matplotlib.pyplot as plt


def fit_agn_model_to_spectral_data(log_freq, log_energy, visualise=True):

    double_skew_guess = [14., .3, 1., 25.5] + [14.5, .3, 1., 25.1]
    initial_guess = double_skew_guess
    fitfunc = lambda p, x: np.log10(agn_models.double_skew_normal_model(x, loc_a=p[0], scale_a=p[1], skew_a=p[2], shift_a=p[3], loc_b=p[4], scale_b=p[5], skew_b=p[6], shift_b=p[7]))

    errfunc = lambda p, x, y: fitfunc(p, x) - y  # y distance in log space
    best_fit, _ = scipy.optimize.leastsq(errfunc, initial_guess, args=(log_freq, log_energy))

    if visualise:
        freq_continuum = np.linspace(log_freq.min(), log_freq.max(), 100)
        plt.plot(log_freq, log_energy, "ro")
        plt.plot(freq_continuum, fitfunc(best_fit, freq_continuum), "r-") # Plot of the data and the fit
        plt.show()

    return best_fit


def fit_host_model_to_spectral_data(log_freq, log_energy, visualise=True):

    initial_guess = [15]  # shift up host, log space

    fitfunc = lambda p, x: agn_models.log_host_flux(x, log_shift=p[0])

    errfunc = lambda p, x, y: fitfunc(p, x) - y  # y distance in log space
    best_fit, _ = scipy.optimize.leastsq(errfunc, initial_guess, args=(log_freq, log_energy))

    # freq_continuum is still in real space, just with log-shaped intervals 
    freq_continuum = np.logspace(np.log10(log_freq.min()), np.log10(log_freq.max()), 100)

    if visualise:
        plt.scatter(log_freq, log_energy, marker='+')
        plt.plot(freq_continuum, fitfunc(best_fit, freq_continuum))
        plt.legend(['Data', 'Best Host Model'])
        plt.show()

    return best_fit


def fit_composite_model_to_spectral_data(log_freq, log_energy, visualise=True):

    initial_guess = [
        .3,  # scale
        1.,  # skew
        25.5,  # shift
        .3,  # scale
        1.,  # skew
        25.1,  # shift
        20  # host shift
    ]

    def double_skew_wrapper(p, x):
        return agn_models.double_skew_normal_model(  # returns in linear space, not the log wrapper
            x, 
            loc_a=14.,  # not free param, fixed
            scale_a=p[0], 
            skew_a=p[1], 
            shift_a=p[2], 
            loc_b=14.5, # not free param, fixed
            scale_b=p[3], 
            skew_b=p[4], 
            shift_b=p[5])

    def host_wrapper(p, x):
        return agn_models.host_flux(x, log_shift=p[6])

    fitfunc = lambda p, x: np.log10(double_skew_wrapper(p, x) + host_wrapper(p, x)) 

    errfunc = lambda p, x, y: fitfunc(p, x) - y  # y distance in log space
    best_fit, success = scipy.optimize.leastsq(errfunc, initial_guess, args=(log_freq, log_energy))

    log_freq_continuum = np.linspace(log_freq.min(), log_freq.max(), 100)

    if visualise:
        plt.scatter(log_freq, log_energy, marker='+')
        plt.plot(log_freq_continuum, fitfunc(best_fit, log_freq_continuum))
        plt.plot(log_freq_continuum, np.log10(host_wrapper(best_fit, log_freq_continuum)))
        plt.plot(log_freq_continuum, np.log10(double_skew_wrapper(best_fit, log_freq_continuum)))
        plt.legend(['Data', 'Combined Model', 'Host Model', 'AGN Model'])
        plt.show()

    return best_fit, success