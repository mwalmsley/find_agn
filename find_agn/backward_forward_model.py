"""
There's a bunch of extreme values in the observed priors,
causing very strange parameter values to be randomly sampled and converted to lines
causing nan line guesses
Presumably this is because the GP is not working right
Some galaxies are not in the model grid, hence GP has no idea, hence gives crazy values

Returns:
    [type]: [description]
"""
import os
from collections import namedtuple

import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from find_agn import bpt_utilities


class MappingsGrid():

    def __init__(self, grid, line_cols, param_cols):
        self.grid = grid
        self.line_cols = line_cols
        self.param_cols = param_cols
        self.ranges = {}
        self.lines_to_params = None
        self.params_to_lines = None

        GridRange = namedtuple('GridRange', ['Min', 'Max'])
        for col in self.grid:
            self.ranges[col] = GridRange._make([grid[col].min(), grid[col].max()])
        
        self.fit()


    def fit(self):
        # use grid to infer relationships between params (e.g. ionization) and emission lines
        self.lines_to_params = fit_lines_to_params(
            self.grid, 
            self.line_cols, 
            self.param_cols
        )
        self.params_to_lines = fit_params_to_lines(
            self.grid, self.line_cols, self.param_cols)


    def describe(self, save_dir):

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # visualise the regression grid (lines_to_params) on BPT diagram
        visualise_lines_to_params(
            self.lines_to_params,
            self.param_cols,
            os.path.join(save_dir, 'lines_to_params.png'),
            grid=self.grid
        )

        # TODO generalise for params_to_lines as well, dif. columns and grid/range
        # hard also because pairs of params do not define predicted line


class DiagnosticModel():

    def __init__(self, grid, galaxies, line_cols, param_cols, mappings_grid):
        self.grid = grid
        self.galaxies = galaxies
        self.line_cols = line_cols
        self.param_cols = param_cols
        self.mappings_grid = mappings_grid
        self.lines_to_params = mappings_grid.lines_to_params
        self.params_to_lines = mappings_grid.params_to_lines
        self.param_prior = None
        self.fitted = False

    def fit(self):
        # use relationships to work out typical params (i.e. param priors)
        self.galaxies = estimate_params_of_galaxies(
            self.galaxies,
            self.lines_to_params,
            self.line_cols,
            self.param_cols)
        self.param_prior = estimate_param_prior(self.galaxies, self.param_cols)
        self.fitted = True

    def forward_pass(self, n_samples=1000):
        # given the prior and the relationships, run a forward model
        # train galaxies must already have estimated params
        # draw from KDE pdf
        assert self.fitted  # must call model.fit() beforehand
        n_initial_samples = n_samples * 1000
        sample_params = np.zeros((n_samples * 1000, len(self.param_cols)))
        for col_n, col, in enumerate(self.param_cols):
            sample_params[:, col_n] = np.random.uniform(
                low=self.galaxies[col].min(),
                high=self.galaxies[col].max(),
                size=n_initial_samples
            )

        sample_params_pdf = self.param_prior(
            sample_params)  # p of each random param draw
        # draw according to pdf
        synthetic_params_index = np.random.choice(
            np.arange(len(sample_params)),
            size=n_samples,
            p=sample_params_pdf/sample_params_pdf.sum()  # must sum to 1 for np to be happy
        )
        synthetic_params = sample_params[synthetic_params_index]
        # calculate lines for those params
        synthetic_lines = self.params_to_lines(synthetic_params)

        synthetic_df = pd.DataFrame(
            data=np.concatenate((synthetic_lines, synthetic_params), axis=1),
            columns=self.line_cols + self.param_cols)

        return synthetic_df

    def describe(self, save_dir):
        """Visualise fitted model
        """
        assert self.fitted
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # describe lines to params, hopefully (but not yet) params to lines
        self.mappings_grid.describe(save_dir)

        # visualise param priors ballparked from real galaxy sample
        g = sns.pairplot(self.galaxies[self.param_cols], diag_kind='kde')
        g.savefig(os.path.join(save_dir, 'prior_approximate_pairplot.png'))

        # TODO not working because pairs of params do not define pdf, all params required
        # leave for now...
        # visualise the actual model priors
        # fig = plt.figure()
        # # for each pair, plot
        # plotted = {}
        # for x_n, param_x in enumerate(self.param_cols):
        #     for y_n, param_y in enumerate(self.param_cols):
        #         if param_x != param_y:
        #             if (param_x, param_y) not in plotted.keys() or (param_y, param_x) not in plotted.keys():
        #                 ax = plt.subplot(len(self.param_cols), len(self.param_cols), len(self.param_cols) * x_n + y_n)  # subplots are 1-indexed
        #                 self.visualise_prior_for_pair(param_x, param_y, ax=ax)
        #                 ax.set_xlabel(param_x)
        #                 ax.set_ylabel(param_y)
        #                 plotted[(param_x, param_y)] = 1
        #                 plotted[(param_y, param_x)] = 1
        # fig.tight_layout()
        # fig.savefig(os.path.join(save_dir, 'prior_exact_flat.png'))

    def visualise_prior_for_pair(self, param_x_name, param_y_name, ax):
        # make a meshgrid
        mesh_axes = []
        n_samples = 10
        for param_n, param in enumerate([param_x_name, param_y_name]):
            mesh_axes.append(np.linspace(
                self.galaxies[param].min(), self.galaxies[param].max(), n_samples))
        meshgrid = np.meshgrid(*mesh_axes)
        # third dim of num. of params
        grouped_params = np.stack(meshgrid, axis=-1)
        pdf = np.zeros([n_samples, n_samples])  # scalar grid

        for i in range(n_samples):
            for j in range(n_samples):
                params_to_measure = np.expand_dims(
                    grouped_params[i, j, :], axis=0)
                prior = self.param_prior(params_to_measure)
                print(prior.shape)
                pdf[i, j] = prior

        ax.contourf(meshgrid[0], meshgrid[1], pdf)


def visualise_lines_to_params(lines_to_params, param_cols, save_loc, grid=None):
    fig, axes = plt.subplots(
        ncols=len(param_cols),
        figsize=(len(param_cols) * 4, 4),
        sharex=True,
        sharey=True
    )

    n_axis_samples = 20
    x_grid, y_grid = np.meshgrid(
        np.linspace(grid['log_NII_HA'].min(),
                    grid['log_NII_HA'].max(), n_axis_samples),
        np.linspace(grid['log_OIII_HB'].min(),
                    grid['log_OIII_HB'].max(), n_axis_samples)
    )
    z = np.zeros((len(x_grid), len(y_grid), len(param_cols)))
    for i in range(n_axis_samples):
        for j in range(n_axis_samples):
            line_pair = np.expand_dims(np.array([x_grid[i, j], y_grid[i, j]]), axis=0)
            assert line_pair.shape == (1, 2)
            param_trio = lines_to_params(line_pair)
            z[i, j, :] = param_trio

    for param_n, param in enumerate(param_cols):
        cs = axes[param_n].contourf(
            x_grid,
            y_grid,
            z[:, :, param_n])
        axes[param_n].set_xlabel('log_NII_HA')
        axes[param_n].set_ylabel('log_OIII_HB')
        axes[param_n].set_title(param)
        fig.colorbar(cs, ax=axes[param_n])

    if grid is not None:
        for ax in axes:
            ax.scatter(
                grid['log_NII_HA'],
                grid['log_OIII_HB'],
                marker='+',
                alpha=0.8)

    fig.tight_layout()
    fig.savefig(save_loc)


def grid_to_bpt_data(loc):

    df = Table.read(loc).to_pandas()

    mapping = {
        'OIII5007': 'OIII_5006',
        'Hbeta': 'HB_4861',
        'Halpha': 'HA_6562',
        'NII6583': 'NII_6583',
        'OI6300': 'OI_6300',
        'SII6716_31': 'SII_6716'
    }

    df = df[list(mapping.keys()) + ['log U', 'log P/k', '12 + log O/H']]
    df = df.rename(columns=mapping)
    df = bpt_utilities.add_bpt_parameters(df)
    df = bpt_utilities.get_galaxy_types(df)
    return df


def multi_output_gaussian_process(df, x_cols, y_cols):    
    x_raw = df[x_cols].values

    scaler_for_x = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler_for_x.fit(x_raw)
    x_scaled = scaler_for_x.transform(x_raw)

    regressors = {}
    scalers_for_y = dict(zip(
        y_cols, 
        [StandardScaler(copy=True, with_mean=True, with_std=True) for n in range(len(y_cols))]
        ))
    saved_alpha = {
        'log U': 2.,  # from 1.5
        'log P/k': 2.,
        '12 + log O/H': 0.1
    }
    for y_col_n, y_col in enumerate(y_cols):
        y_raw = df[y_col].values.reshape(-1, 1)
        scalers_for_y[y_col].fit(y_raw)
        y_scaled = scalers_for_y[y_col].transform(y_raw)

        # Matern kernel is smooth-ish with extra free param, crucial for good fit
        # alpha determines X uncertainty i.e. regularization
        # currently picked by eye because grid is very uneven, not sure how to improve
        if y_col in saved_alpha.items():
            alpha = saved_alpha[y_col]
        else:
            alpha = 0.5
        regressor = GaussianProcessRegressor(
            n_restarts_optimizer=10, kernel=Matern(), alpha=alpha)

        regressor.fit(x_scaled, y_scaled)  # set up to predict that column
        regressors[y_col] = regressor

    saved_sigma_clips = {    # similarly by eye :(
        'log U': 0.6,
        'log P/k': 0.7,  # from 0.65
        '12 + log O/H': 0.3
    }
    def clipped_predict(x):  # using closure to pass in regressors, scalers
        y = np.zeros((len(x), len(y_cols)))
        for y_col_n, y_col in enumerate(y_cols):
            x_scaled = scaler_for_x.transform(x)

            y_single_scaled, sigma = regressors[y_col].predict(x_scaled, return_std=True)
            # set uncertain values to nan, if there's a saved max uncertainty
            if y_col in saved_sigma_clips.keys():
                max_sigma = saved_sigma_clips[y_col]
                y_single_scaled[sigma > max_sigma] = np.nan
            y_single = scalers_for_y[y_col].inverse_transform(y_single_scaled)
            y[:, y_col_n] = y_single.squeeze()
        return y

    return clipped_predict


def fit_lines_to_params(df, line_cols, param_cols):
    return multi_output_gaussian_process(df, x_cols=line_cols, y_cols=param_cols)


def fit_params_to_lines(df, line_cols, param_cols):
    return multi_output_gaussian_process(df, x_cols=param_cols, y_cols=line_cols)


def estimate_params_of_galaxies(input_df, regressor, line_cols, param_cols):
    # Wrapper. Useful for debugging regressor, and for science.
    df = input_df.copy()  # no side effects
    X = df[line_cols].dropna(how='any')
    predictions = regressor(X)  # lines to params

    for col_n, col in enumerate(param_cols):
        df[col] = predictions[:, col_n]

    return df


def estimate_param_prior(df, param_cols):
    data = df[param_cols].values
    params_kde = KDEMultivariate(
        data,
        var_type=''.join(['c'] * len(param_cols))
        # bw='cv_ml' Temporarily disable cv bandwidth selection for speed
    )
    return lambda x: params_kde.pdf(x)


if __name__ == '__main__':
    line_cols = ['log_NII_HA', 'log_OIII_HB']
    param_cols = ['log U', 'log P/k', '12 + log O/H']
    grid_loc = os.path.join('find_agn/tests/test_examples/NB_HII_grid.fits')
    # all grid points are useful so include all
    grid = grid_to_bpt_data(grid_loc).dropna(how='any', subset=line_cols)
    mappings_grid = MappingsGrid(grid, line_cols, param_cols)
    mappings_grid.describe('figures/first_grid')

    galaxies = pd.read_csv(
        'bpt_df.csv', nrows=1000)
    galaxies = bpt_utilities.normalize_to_hbeta(galaxies)

    # only include BPT-region starforming galaxies for P
    galaxies = galaxies[galaxies['galaxy_type'] == 'starforming']
    galaxies = bpt_utilities.zoom_bpt_diagram(galaxies)

    model = DiagnosticModel(grid, galaxies, line_cols, param_cols, mappings_grid)
    model.fit()
    model.describe('figures/first_model')

    synthetic_df = model.forward_pass(100)
    synthetic_df.to_csv('figures/first_model/forward_pass.csv')
    fig, ax = plt.subplots()
    sns.kdeplot(synthetic_df['log_NII_HA'], synthetic_df['log_OIII_HB'], ax=ax)
    ax.set_xlabel('log_NII_HA')
    ax.set_ylabel('log_OIII_HB')
    ax.set_title('Synthetic Galaxies')
    fig.tight_layout()
    fig.savefig('figures/first_model/forward_pass.png')

    fig, ax = plt.subplots()
    sns.kdeplot(galaxies['log_NII_HA'], galaxies['log_OIII_HB'], ax=ax, shade_lowest=False)
    sns.kdeplot(synthetic_df['log_NII_HA'], synthetic_df['log_OIII_HB'], ax=ax, shade_lowest=False)
    ax.set_xlim([-1.5, 0.5])
    ax.set_ylim([-2, 1])
    fig.savefig('figures/first_model/forward_pass_comparison.png')
