import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from find_agn import spectral_fitting


class Continuum():

    def __init__(self, freq, flux_density, background=None, line_data=None):
     # templates are abstract - once given a specific z (time), becomes specific continuum

        self.frequency = freq  # Hz
        self.flux_density = flux_density

        # TODO is there a way to dynamically recalculate properties when called without using funcs?
        self.wavelength = 299792458 / self.frequency  # lambda (m) = c (ms^1) / freq (Hz)

        # energy density v Fv i.e. 10^-23 erg / s / cm^2
        self.energy_density = self.flux_density * self.frequency

        if background is None:
            self.background = None
            self.flux_density_subtracted = None
            self.background_func = None
        else:
            self.background = background
            self.flux_density_subtracted = self.background - self.flux_density

        self.line_data = line_data


    def get_spectral_data(self):  # for now, continue using pandas not custom object for fitting
        df = pd.DataFrame(data={
            'frequency': self.frequency,
            'wavelength': self.wavelength,
            'flux_density': self.flux_density,
            'energy_density': self.energy_density
            })

        if self.background is not None:
            df['continuum_flux_density'] = self.background
            df['flux_density_subtracted'] = self.background - self.flux_density

        return df


    def fit_background(self, **kwargs):
        # pass any kwargs forward to the fitter e.g. included, excluded
        if self.background is not None:
            raise ValueError('Attempting to fit explicitly-provided background')
        # interface with spectral_fitting via pandas
        data_with_bkg = spectral_fitting.fit_background(self.get_spectral_data(), **kwargs)
        self.background = data_with_bkg['continuum_flux_density'].values
        self.flux_density_subtracted = self.background - self.flux_density


    def get_bpt_lines(self):
        # interface with spectral_fitting via pandas
        if self.flux_density_subtracted is None:
            self.fit_background(smoothing_frac=0.03)  # good general performance - but SENSITIVE!

        # quite messy with spectral_fitting
        spec_object = spectral_fitting.load_spectrum(
            self.wavelength * 1e10,  # fitter expects angstroms
            self.flux_density_subtracted)

        spectral_fitting.fit_lines(spec_object)  # inplace
        # modifies spec_object inplace and returns data
        self.line_data = spectral_fitting.measure_line_properties(spec_object)
        self.spec_object = spec_object  # save for plotting of fit. Note: angstroms!


    def measured_bpt_lines(self):
        return line_data_to_bpt_flux(self.line_data)


    def visualise_spectrum(self):
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 6))
        data = self.get_spectral_data()
        data['wavelength'] = data['wavelength'] * 1e10  # view in angstroms
        df_ha = data[(data['wavelength'] > 6450) & (data['wavelength'] < 6775)]
        df_hb = data[(data['wavelength'] > 4830) & (data['wavelength'] < 5100)]
        for (df, ax) in [(df_ha, ax0), (df_hb, ax1)]:
            ax.plot(df['wavelength'], df['flux_density'], label='emission')
            ax.plot(df['wavelength'], df['continuum_flux_density'], label='background')
            ax.set_xlabel('Wavelength (A)')
            ax.set_ylabel('Flux density')
        return fig, (ax0, ax1)


    def visualise_fit(self):
        fig, (dummy_ax0, dummy_ax1) = plt.subplots(nrows=2)
        spectral_fitting.plot_lines_in_range(self.spec_object.copy(), fig, dummy_ax0, 6450, 6775)
        spectral_fitting.plot_lines_in_range(self.spec_object.copy(), fig, dummy_ax1, 4830, 5100)
        return fig  # no ax to return?


def line_data_to_bpt_flux(line_data):

    data = {}

    lines = {
        'OIIIb': 'OIII_5006',
        'H_beta': 'HB_4861',
        'NIIb': 'NII_6583',
        'SIIa': 'SII_6716',  # or b?
        'OI': 'OI_6300',
    }

    for old_name, bpt_name in lines.items():
        line = line_data[line_data['line'] == old_name].squeeze()
        assert isinstance(line, pd.Series)
        data[bpt_name] = line['flux']  # or amplitude? Note: can't simply add amplitudes

    # add H alpha
    h_alpha_flux = line_data[line_data['line'].isin(['H_alpha', 'H_alpha_1'])]['flux'].sum()
    data['HA_6562'] = h_alpha_flux

    return pd.Series(data)
