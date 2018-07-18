import pandas as pd
from astropy import units

import matplotlib
matplotlib.use('Agg')

import pyspeckit


def load_spectrum(wavelength, flux):
    # Initialize spectrum object
    wavelength_unit = units.AA
    flux_unit = units.erg/units.s/units.cm**2/units.AA
    spec = pyspeckit.Spectrum(
        data=flux, 
        xarr=wavelength,
        xarrkwargs={'unit': wavelength_unit},
        unit=flux_unit)
    return spec


def fit_lines(spec):

    # Rest wavelengths of the lines we are fitting - use as initial guesses
    NIIa = 6549.86
    NIIb = 6585.27
    Halpha = 6564.614
    Hbeta = 4861.00
    SIIa = 6718.29
    SIIb = 6732.68


    # We fit the [NII] and [SII] doublets, and allow two components for Halpha.
    # The widths of all narrow lines are tied to the widths of [SII].
    guesses = [  # gaussian param. initial guesses per line in form amplitude, center, width
        50, NIIa, 5, 
        100, Halpha, 5, 
        50, Halpha, 50,   # two components for Halpha
        1, Hbeta, 3,
        50, NIIb, 5, 
        20, SIIa, 5, 
        20, SIIb, 5  # final param (17) is width for width limits
    ]

    tied = [   # width param (3rd col) tied to width of SIIb, param 17
        '', '', 'p[-1]',  # NIIa
        '', '', 'p[-1]',  # Halpha 1
        '', 'p[4]', '',  # Halpha 2
        '', '', '',  # unconstrained Hbeta, doesn't fit properly yet
        '3 * p[0]', '', 'p[-1]', 
        '', '', 'p[-1]', 
        '', '', ''
    ]

    # Actually do the fit
    spec.specfit(guesses = guesses, tied = tied, annotate = False)


def measure_line_properties(spec):
    # Let's use the measurements class to derive information about the emission
    # lines.  The galaxy's redshift and the flux normalization of the spectrum
    # must be supplied to convert measured fluxes to line luminosities.
    spec.measure(z = 0, fluxnorm = 1e-12)

    lines = []
    for line in spec.measurements.lines.keys():
        lines.append({
            'line': line,
            'flux': spec.measurements.lines[line]['flux'],
            'amplitude': spec.measurements.lines[line]['amp'],
            'fwhm': spec.measurements.lines[line]['fwhm'],
            'luminosity': spec.measurements.lines[line]['lum']
        })
    return pd.DataFrame(data=lines)


def overplot_measured_lines(spec):
    # Now overplot positions of lines and annotate
    y = spec.plotter.ymax * 0.85    # Location of annotations in y

    for i, line in enumerate(spec.measurements.lines.keys()):

        # If this line is in our database of lines, annotate it
        if line in spec.speclines.optical.lines.keys():

            x = spec.measurements.lines[line]['modelpars'][1]   # Location of the emission line
            # Draw dashed line to mark its position
            spec.plotter.axis.plot([x]*2, [spec.plotter.ymin, spec.plotter.ymax],
                                ls='--', color='k')
            # Label it
            spec.plotter.axis.annotate(spec.speclines.optical.lines[line][-1], (x, y),
                                    rotation = 90, ha = 'right', va = 'center')
    # Make some nice axis labels
    spec.plotter.axis.set_xlabel(r'Wavelength $(\AA)$')
    spec.plotter.axis.set_ylabel(r'Flux $(10^{-17} \mathrm{erg/s/cm^2/\AA})$')
    spec.plotter.refresh()


if __name__ == '__main__':

    optical_spectra = pd.read_csv('optical_spectra.csv')

    wavelength = optical_spectra['wavelength'].values
    assert wavelength.mean() < 10000 and wavelength.mean() > 1000  # should be in angstroms
    flux = optical_spectra['energy'].values * 1e-12 / wavelength  # expects a flux, normalise to 1e-12

    spec = load_spectrum(wavelength, flux)
    # xmin = 6450
    # xmax = 6775
    # xmin = wavelength.min()
    # xmax = wavelength.max()
    xmin = 4841.00
    xmax = 4921.00

    # plot region surrounding Halpha-[NII] complex
    # spec.plotter(xmin=xmin, xmax=xmax, ymin=0, ymax=250)
    spec.plotter(xmin=xmin, xmax=xmax)
    # spec.plotter()

    fit_lines(spec)  # inplace

    spec.plotter.refresh()

    line_data = measure_line_properties(spec)  # inplace

    overplot_measured_lines(spec)  # inplace

    spec.specfit.plot_fit()

    spec.plotter.figure.savefig("sdss_fit_example.png")
    line_data.to_csv('line_data.csv', index=False)

# TODO good fits to the pre-built quantities but not to Hbeta. Need to get Hbeta working.