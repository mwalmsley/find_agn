import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import pytest

from find_agn import spectral_fitting
from find_agn.tests import TEST_FIGURE_DIR 

@pytest.fixture()
def raw_spectral_data():
    wavelength = np.linspace(0, 10000, num=10000)
    continuum = wavelength * 0.01 + 50

    stddev = 5
    noise = np.random.randn(wavelength.size) * stddev

    emission = np.zeros_like(continuum)
    for line in spectral_fitting.TARGET_LINES:
        emission += 200 * scipy.stats.norm.pdf(
            wavelength,
            loc=line['wavelength'],
            scale=5.)

    flux_density = continuum + emission + noise

    # plt.clf()
    # plt.plot(wavelength, flux_density)
    # plt.savefig(TEST_FIGURE_DIR + '/reference_spectrum_raw.png')

    return pd.DataFrame(data={
        'wavelength': wavelength,
        'flux_density': flux_density})


@pytest.fixture()
def prepared_spectral_data():
    wavelength = np.linspace(4500, 7000, num=10000)
    continuum = wavelength * 0.01 + 50

    stddev = 5
    noise = np.random.randn(wavelength.size) * stddev

    emission = np.zeros_like(continuum)
    for line in spectral_fitting.TARGET_LINES:
        emission += 200 * scipy.stats.norm.pdf(
            wavelength,
            loc=line['wavelength'],
            scale=5.)

    flux_density = continuum + emission + noise

    plt.clf()
    plt.plot(wavelength, flux_density)
    plt.savefig(TEST_FIGURE_DIR + '/reference_spectrum_prepared.png')

    return pd.DataFrame(
        data={
            'wavelength': wavelength,
            'flux_density': flux_density,
            'continuum_flux_density': continuum,
            'flux_density_subtracted': continuum - flux_density
            }
        )


@pytest.fixture()
def spec_object(prepared_spectral_data):
    # NB: we do not test the loading itself
    return spectral_fitting.load_spectrum(
        prepared_spectral_data['wavelength'].values,
        prepared_spectral_data['flux_density_subtracted'].values
        )


def test_fit_lowess_continuum(raw_spectral_data):
    continuum_fit = spectral_fitting.fit_lowess_continuum(raw_spectral_data)
    continuum = continuum_fit(raw_spectral_data['wavelength'])
    assert np.max(np.abs(continuum - raw_spectral_data['flux_density'])) < 100
    plt.plot(raw_spectral_data['wavelength'], raw_spectral_data['flux_density'])
    plt.plot(raw_spectral_data['wavelength'], continuum)
    plt.savefig(TEST_FIGURE_DIR + '/continuum.png')



def test_prepare_spectral_data(raw_spectral_data):
    data = spectral_fitting.prepare_spectral_data(raw_spectral_data)
    assert data['continuum_flux_density'].mean() < data['flux_density'].max()
    assert data['continuum_flux_density'].mean() > data['flux_density'].min()
    assert np.abs(data['flux_density_subtracted']).mean() < 100


def test_measure_line_properties(spec_object):
    spectral_fitting.fit_lines(spec_object)  # inplace
    line_data = spectral_fitting.measure_line_properties(spec_object)
    line_data.to_csv(TEST_FIGURE_DIR + '/line_data.csv')

    # fig = plt.figure()
    fig, (dummy_ax0, dummy_ax1) = plt.subplots(nrows=2)
    spectral_fitting.plot_lines_in_range(spec_object.copy(), fig, dummy_ax0, 6450, 6775)
    spectral_fitting.plot_lines_in_range(spec_object.copy(), fig, dummy_ax1, 4830, 5100)
    # fig.tight_layout()
    fig.savefig(TEST_FIGURE_DIR + '/measured_line_properties.png')
    plt.clf()

    for line in spectral_fitting.TARGET_LINES:
        measurements = line_data[line_data['line'] == line['name']].squeeze()
        assert isinstance(measurements, pd.Series)
        assert (measurements['amplitude'].max() < -10) and (measurements['amplitude'].min() > -20)
