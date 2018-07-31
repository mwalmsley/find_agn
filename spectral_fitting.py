"""Fit galaxy spectra to find emission lines
"""


import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from astropy import units
import pyspeckit
from scipy.interpolate import interp1d

from scratch import copy_to_dummy_ax


TARGET_LINES = [
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


def find_emission_lines(raw_spectral_data, data_loc=None, figure_loc=None):

    spectral_data = prepare_spectral_data(raw_spectral_data, normalisation=1e24)

    spec_object = load_spectrum(
        spectral_data['wavelength'].values,
        spectral_data['flux_density_subtracted'].values)

    fit_lines(spec_object)  # inplace
    line_data = measure_line_properties(spec_object)  # inplace and returns data

    if data_loc is not None:
        line_data.to_csv(data_loc, index=False)

    if figure_loc is not None:
        fig, (axes_upper, axes_lower) = plt.subplots(nrows=2, ncols=2)
        visualise_continuum(spectral_data, axes_upper[0], axes_upper[1])

        plot_lines_in_range(spec_object.copy(), fig, axes_lower[0], 6450, 6775)
        plot_lines_in_range(spec_object.copy(), fig, axes_lower[1], 4830, 5100)
        fig.savefig(os.path.join(figure_dir, debug_figure_loc))

    # TODO cannot save tight layout figure because copied axes are not truly subplots
    # They are simply set to the same location with get_position()
    # Tight layout therefore moves both copied axes to fill the full figure
    # fig.tight_layout()
    # fig.savefig(os.path.join(figure_dir, 'fit_debug_details_tight.png'))

    return line_data


def prepare_spectral_data(spectrum, normalisation=1.):
    """
    Normalise and trim spectrum
    Estimate continuum emission (excluding Ha) and save as 'continuum_flux_density'
    Subtract raw flux from continuum emission and save as 'flux_density_subtracted'

    Args:
        spectrum (pd.DataFrame): spectral data, form {wavelength: np.array, flux_density: np.array}

    Returns:
        pd.DataFrame: rescaled/trimmed spectral data with continuum flux estimated and subtracted
    """
    wavelength = spectrum['wavelength'].values
    assert wavelength.mean() < 10000 and wavelength.mean() > 1000  # should be in angstroms

    # normalise to sane absolute values
    spectrum['flux_density'] = spectrum['flux_density'] * normalisation
    assert np.max(spectrum['flux_density']) < 1e5
    assert np.min(spectrum['flux_density']) > 1e-5

    xmin = 4830
    xmax = 6800
    spectrum = spectrum[
        (spectrum['wavelength'] > xmin) &
        (spectrum['wavelength'] < xmax)
    ]

    # fit continuum from spectrum, excluding the Ha region
    continuum = fit_lowess_continuum(spectrum, excluded=[(6562-10, 6562+10)])

    spectrum['continuum_flux_density'] = continuum(spectrum['wavelength'])
    spectrum['flux_density_subtracted'] = spectrum['continuum_flux_density'] - \
        spectrum['flux_density']

    return spectrum


def fit_lowess_continuum(input_spec, filter_range=None, excluded=None):
    """Estimate the continuum emission of a spectra using local regression.

    Args:
        input_spec (pd.DataFrame): of form {wavelength: np.array, flux_density: np.array}
        filter_range (tuple, optional): Defaults to None. Min and max range for spectra to fit
        excluded (list, optional): Defaults to None. Tuple pairs of limits to exclude from fit

    Returns:
        function: continuous model of spectral continuum (i.e. 'background' of lines)
    """

    if filter_range is None:
        spec = input_spec.copy()
    else:
        spec = input_spec[
            (input_spec['wavelength'] > filter_range[0]) &
            (input_spec['wavelength'] < filter_range[1])
        ]

    spec['masked'] = False
    if excluded is not None:
        for wavelength_pair in excluded:
            spec['masked'] = spec['masked'] | ((spec['wavelength'] > wavelength_pair[0]) & (
                spec['wavelength'] < wavelength_pair[1]))

    lowess = sm.nonparametric.lowess

    spec_to_fit = spec[~spec['masked']]
    spec_to_fit = spec_to_fit.sort_values('wavelength')

    continuum = lowess(
        spec_to_fit['flux_density'], spec_to_fit['wavelength'], is_sorted=True, frac=.12)
    continuum_model = interp1d(
        continuum[:, 0], continuum[:, 1], kind='quadratic')

    return continuum_model


def visualise_continuum(spectrum, continuum_ax, subtracted_ax):
    """[summary]

    Args:
        spectrum ([type]): [description]
        continuum_ax ([type]): [description]
        subtracted_ax ([type]): [description]
    """

    # plot spectra and best fit
    continuum_ax.plot(
        spectrum['wavelength'],
        spectrum['flux_density'],
        'k--',
        label='Actual')
    # continuum_ax.plot
    # (continuum[:, 0], continuum[:, 1], 'r--', label='est. continuum', label='continuum')
    continuum_ax.plot(
        spectrum['wavelength'],
        spectrum['continuum_flux_density'],
        'r',
        label='Continuum')
    continuum_ax.legend()
    continuum_ax.set_ylabel('Flux Density')

    subtracted_ax.plot(
        spectrum['wavelength'],
        spectrum['flux_density_subtracted'],
        label='Continuum - Actual')
    subtracted_ax.legend()
    subtracted_ax.set_ylabel('Flux Density')


def load_spectrum(wavelength, flux):
    """[summary]

    Args:
        wavelength ([type]): [description]
        flux ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Initialize spectrum object
    wavelength_unit = units.AA
    flux_unit = units.erg/units.s/units.cm**2/units.AA
    return pyspeckit.Spectrum(
        data=flux,
        xarr=wavelength,
        xarrkwargs={'unit': wavelength_unit},
        unit=flux_unit)


def fit_lines(spec):
    """Find BPT diagram lines in spectral object. Save best fit inplace to object.

    Args:
        spec (pyspeckit.spectrum): spectral object to fit and modify inplace
    """
    # Rest wavelengths of the lines we are fitting - use as initial guesses
    # TODO refactor to ensure consistency with TARGET_LINES while keeping guesses concise
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
        20., Hbeta, 5,  # maybe width of .3
        20, OI, 5,
        20, OIII, 5,
        5, NII, 1,
        100, Halpha, 5,
        50, Halpha, 50,   # two components for Halpha
        20, SII, 5,
        20, SIIb, 5  # final param is width for width limits
    ]

    tied = [   # width param (3rd col) tied to width of SIIb, param 17
        '', 'p[1]', 'p[-1]',  # unconstrained Hbeta (maybe constrained)
        '', 'p[4]', 'p[-1]',  # OI
        '', 'p[7]', 'p[-1]',  # OIII
        '', 'p[10]', 'p[-1]',  # NIIa
        '', 'p[13]', 'p[-1]',  # Halpha 1
        '', 'p[13]', '',  # Halpha 2
        '', 'p[19]', 'p[-1]',  # SII
        '', 'p[22]', ''  # unconstrained width SIIb, used to constrain some others
    ]

    # Actually do the fit. Populate spec.modelpars.
    # tied is passed as kwarg through specfit to fitter and then to gaussian fit
    spec.specfit(guesses=guesses, tied=tied, annotate=False)


def measure_line_properties(spec):
    """Derive information about the emission lines.

    The galaxy's redshift and the flux normalization of the spectrum
    must be supplied to convert measured fluxes to line luminosities.

    Args:
        spec ([type]): [description]

    Returns:
        [type]: [description]
    """
    # use spec.modelpars to fill spec.measurements
    spec.measure(z=0., miscline=None, ptol=3., misctol=3.)

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


def plot_lines_in_range(spec, target_fig, dummy_ax, xmin, xmax):
    """[summary]

    Args:
        spec ([type]): [description]
        ax ([type]): [description]
        xmin ([type]): [description]
        xmax ([type]): [description]
    """
    spec.plotter(xmin=xmin, xmax=xmax)
    spec.plotter.refresh()
    overplot_measured_lines(spec)  #  inplace
    spec.specfit.plot_fit(annotate=False)
    copy_to_dummy_ax(spec.plotter.axis, dummy_ax, target_fig)


def overplot_measured_lines(spec):
    """overplot positions of lines and annotate

    Args:
        spec ([type]): [description]
    """
    y = spec.plotter.ymax * 0.85    # Location of annotations in y
    for line in spec.measurements.lines.keys():
        # If this line is in our database of lines, annotate it
        if line in spec.speclines.optical.lines.keys():
            # Location of the emission line
            x = spec.measurements.lines[line]['modelpars'][1]
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
