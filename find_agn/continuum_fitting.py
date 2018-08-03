import matplotlib
matplotlib.use('Agg')

import numpy as np
import pyspeckit

def make_synthetic_spectra_with_continuum(xaxis, sigma, center):

    baseline = np.poly1d([0.1, 0.25])(xaxis)

    synth_data = np.exp(-(xaxis-center)**2/(sigma**2 * 2.)) + baseline

    # Add noise
    stddev = 0.1
    noise = np.random.randn(xaxis.size)*stddev
    error = stddev*np.ones_like(synth_data)
    data = noise+synth_data

    # this will give a "blank header" warning, which is fine
    return pyspeckit.Spectrum(data=data, error=error, xarr=xaxis,
                            xarrkwargs={'unit':'km/s'},
                            unit='erg/s/cm^2/AA')


def fit_polynomial_continuum(spec, order=2, exclude=[6550, 6565]):  # inplace
    """https://github.com/pyspeckit/pyspeckit/blob/master/pyspeckit/spectrum/models/polynomial_continuum.py
    """

    spec.specfit.Registry.add_fitter('polycontinuum',
                                pyspeckit.models.polynomial_continuum.poly_fitter(order=order), 
                                2)

    guesses = np.zeros(order + 1)  # need intercept guess
    spec.specfit(fittype='polycontinuum', guesses=guesses, exclude=exclude)


if __name__ == '__main__':

    xaxis = np.linspace(-50,150,100.)
    sigma = 10.
    center = 50.
    sp = make_synthetic_spectra_with_continuum(xaxis, sigma, center)

    sp.plotter()  # add plotting object
    sp.plotter.figure.savefig('cont.png')

    fit_polynomial_continuum(sp)

    # subtract the model fit to create a new spectrum
    sp_contsub = sp.copy()  # don't modify original spectrum object, copy first
    sp_contsub.data -= sp.specfit.get_full_model()  
    sp_contsub.plotter()
    
    # Fit with automatic guesses
    sp_contsub.specfit(fittype='gaussian')

    # Fit with input guesses
    # The guesses initialize the fitter
    # This approach uses the 0th, 1st, and 2nd moments
    data = sp_contsub.data
    amplitude_guess = data.max()
    center_guess = (data*xaxis).sum()/data.sum()
    width_guess = (data.sum() / amplitude_guess / (2*np.pi))**0.5
    guesses = [amplitude_guess, center_guess, width_guess]
    sp_contsub.specfit(fittype='gaussian', guesses=guesses)

    sp_contsub.plotter(errstyle='fill')
    sp_contsub.specfit.plot_fit()

    sp_contsub.plotter.figure.savefig('cont_sub.png')
