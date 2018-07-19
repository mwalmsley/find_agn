import os

import pandas as pd
from astropy import units

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pyspeckit
from scipy.interpolate import interp1d

from continuum_fitting import fit_polynomial_continuum


target_lines = [
        {
            'name': 'H_beta',
            'wavelength': 4861.
        }, 
        {
            'name': 'OIII',
            'wavelength': 5006.
        }, 
        {
            'name': 'OI',
            'wavelength': 6300.
        }, 
        {
            'name': 'NII',
            'wavelength': 6583.
        }, 
        {
            'name': 'H_alpha',
            'wavelength': 6562.
        },
        {
            'name': 'SII',
            'wavelength': 6716.
        }, 
        {
            'name': 'SIIb',
            'wavelength': 6732.68
        }
    ]


def fit_lowess_continuum(input_spec, allowed=None, excluded=None, visualise=False):

    if allowed is None:
        spec = input_spec.copy()
    else:
        spec = input_spec[
            (input_spec['wavelength'] > allowed[0]) &
            (input_spec['wavelength'] < allowed[1])
        ]
        
    spec['masked'] = False
    if excluded is not None:
        for wavelength_pair in excluded:
            spec['masked'] = spec['masked'] | ((spec['wavelength'] > wavelength_pair[0]) & (spec['wavelength'] < wavelength_pair[1]))

    lowess = sm.nonparametric.lowess

    spec_to_fit = spec[~spec['masked']]
    spec_to_fit = spec_to_fit.sort_values('wavelength')

    continuum = lowess(spec_to_fit['flux_density'], spec_to_fit['wavelength'], is_sorted=True, frac=.12)
    continuum_model = interp1d(continuum[:, 0], continuum[:, 1], kind='quadratic')

    if visualise:
        sample_wavelengths = np.linspace(spec['wavelength'].min(), spec['wavelength'].max(), 1000)
        plt.plot(spec['wavelength'], spec['flux_density'], 'k--')
        plt.plot(continuum[:, 0], continuum[:, 1], 'r--', label='est. continuum')
        plt.plot(sample_wavelengths, continuum_model(sample_wavelengths), 'r', label='model')

    return continuum_model



def load_spectrum(wavelength, flux):
    # Initialize spectrum object
    wavelength_unit = units.AA
    flux_unit = units.erg/units.s/units.cm**2/units.AA
    return pyspeckit.Spectrum(
        data=flux, 
        xarr=wavelength,
        xarrkwargs={'unit': wavelength_unit},
        unit=flux_unit)


def fit_lines(spec):



    # Rest wavelengths of the lines we are fitting - use as initial guesses
    Hbeta = 4861.
    OIII = 5006
    OI = 6300
    NII = 6583.
    Halpha = 6562.
    SII = 6716.
    SIIb = 6732.68


    # We fit the [NII] and [SII] doublets, and allow two components for Halpha.
    # The widths of all narrow lines are tied to the widths of [SII].
    guesses = [  # gaussian param. initial guesses per line in form amplitude, center, width
        20., Hbeta, .3,
        20, OI, 5,
        20, OIII, 5,
        5, NII, 1, 
        100, Halpha, 5, 
        50, Halpha, 50,   # two components for Halpha
        20, SII, 5,
        20, SIIb, 5  # final param is width for width limits
    ]

    tied = [   # width param (3rd col) tied to width of SIIb, param 17
        '', 'p[1]', '',  # unconstrained Hbeta
        '', 'p[4]', 'p[-1]',  # OI
        '', 'p[7]', 'p[-1]',  # OIII
        '', 'p[10]', 'p[-1]',  # NIIa
        '', 'p[13]', 'p[-1]',  # Halpha 1
        '', 'p[13]', '',  # Halpha 2
        '', 'p[19]', 'p[-1]',  # SII
        '', 'p[22]', ''  # unconstrained width SIIb, used to constrain some others
    ]

    # Actually do the fit. Populate spec.modelpars.
    spec.specfit(guesses=guesses, tied=tied, annotate=False)  # tied as passed as kwarg through specfit to fitter and then to gaussian fit


def measure_line_properties(spec):
    """Let's use the measurements class to derive information about the emission
    lines.  The galaxy's redshift and the flux normalization of the spectrum
    must be supplied to convert measured fluxes to line luminosities.

    Args:
        spec ([type]): [description]

    Returns:
        [type]: [description]
    """
    # spec.measure()
   
    spec.measure(z = 0., miscline=None, ptol=3., misctol=3.)  # use spec.modelpars to fill spec.measurements

    lines = []
    for line in spec.measurements.lines.keys():
        lines.append({
            'line': line,
            'flux': spec.measurements.lines[line]['flux'],
            'amplitude': spec.measurements.lines[line]['amp'],
            'fwhm': spec.measurements.lines[line]['fwhm'],
            'luminosity': spec.measurements.lines[line]['lum'],
            'wavelength': spec.measurements.lines[line]['pos']
        })
    return pd.DataFrame(data=lines)


def overplot_measured_lines(spec):
    """overplot positions of lines and annotate
    
    Args:
        spec ([type]): [description]
    """
    y = spec.plotter.ymax * 0.85    # Location of annotations in y
    for i, line in enumerate(spec.measurements.lines.keys()):
        # If this line is in our database of lines, annotate it
          if line in spec.speclines.optical.lines.keys():
            # measurements.lines[line]['modelpars'] copied by spec.measure from modelpars found in fit 
            x = spec.measurements.lines[line]['modelpars'][1]   # Location of the emission line
            # Draw dashed line to mark its position
            spec.plotter.axis.plot(
                [x]*2, 
                [spec.plotter.ymin, spec.plotter.ymax],
                 ls='--', 
                 color='k')
            # Label it
            spec.plotter.axis.annotate(
                spec.speclines.optical.lines[line][-1], 
                (x, y),
                rotation=90, 
                ha='right', 
                va='center')
    # Make some nice axis labels
    spec.plotter.axis.set_xlabel(r'Wavelength $(\AA)$')
    spec.plotter.axis.set_ylabel(r'Flux $(10^{-17} \mathrm{erg/s/cm^2/\AA})$')
    spec.plotter.refresh()


def save_plot(spec, xmin, xmax, save_loc):
    spec.plotter(xmin=xmin, xmax=xmax)
    spec.plotter.refresh()
    overplot_measured_lines(spec)  # inplace
    spec.specfit.plot_fit(annotate=False)
    spec.plotter.figure.savefig(save_loc)


if __name__ == '__main__':

    spectra = pd.read_csv('optical_spectra.csv')

    wavelength = spectra['wavelength'].values
    assert wavelength.mean() < 10000 and wavelength.mean() > 1000  # should be in angstroms

    # normalise to 1e-12
    spectra['flux_density'] = spectra['flux_density'] * 1e24

    xmin = 4830
    xmax = 6800
    # xmin = wavelength.min()
    # xmax = wavelength.max()

    # restrict to Ha region
    spectra = spectra[
        (spectra['wavelength'] > xmin) &
        (spectra['wavelength'] < xmax)
    ]

    continuum_model = fit_lowess_continuum(spectra, excluded=[(6562-10, 6562+10)], visualise=True)

    figure_dir = 'spectral_fits/reference_model'
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)

    plt.savefig(os.path.join(figure_dir, 'continuum.png'))
    plt.clf()

    spectra['continuum_flux_density'] = continuum_model(spectra['wavelength'])
    spectra['flux_density_subtracted'] =  spectra['continuum_flux_density'] - spectra['flux_density']

    plt.plot(spectra['wavelength'], spectra['flux_density_subtracted'])
    plt.savefig(os.path.join(figure_dir, 'continuum_subtracted.png'))

    spec = load_spectrum(spectra['wavelength'].values, spectra['flux_density_subtracted'].values)
    # spec = pyspeckit.Spectrum('sample_sdss.txt', errorcol=2)

    fit_lines(spec)  # inplace
    line_data = measure_line_properties(spec)  # inplace and returns data
    line_data.to_csv('line_data.csv', index=False)

    save_plot(spec, xmin, xmax, os.path.join(figure_dir, 'sdss_fit_full.png'))
    save_plot(spec, 6450, 6775, os.path.join(figure_dir, 'sdss_fit_near_ha.png'))
    save_plot(spec, 4830, 5100, os.path.join(figure_dir, 'sdss_fit_near_hb_O3.png'))
