import os
from copy import copy

import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from find_agn import backward_forward_model, bpt_utilities
from find_agn.tests import TEST_EXAMPLE_DIR, TEST_FIGURE_DIR


@pytest.fixture()
def grid_loc():
    # copied from OXAF
    return os.path.join(TEST_EXAMPLE_DIR, 'NB_HII_grid.fits')
    # agn_nlr_grid_loc = 'data/nebulabayes/NB_NLR_grid.fits'  # TODO sensitive to current loc


@pytest.fixture()
def line_cols():
    return ['log_NII_HA', 'log_OIII_HB']


@pytest.fixture()
def param_cols():
    return ['ionization', 'pressure', 'another']


@pytest.fixture()
def params_to_lines(line_cols):
    def params_to_lines_callable(param_values):  # normal arg, not fixture
        line_pure_values = np.sum(param_values, axis=1)
        line_sim_noise = 0.5 * np.random.normal(size=line_pure_values.size)
        line_values = line_pure_values + line_sim_noise  # simulated_params
        # add params by column for expected line(s?)
        return np.stack([line_values] * len(line_cols), axis=1)
    return params_to_lines_callable


@pytest.fixture()
def lines_to_params(param_cols):
    # implicit match required with params_to_lines
    def lines_to_params_callabe(line_values):
        param_values = line_values.mean(
            axis=1) / len(param_cols)  # sum of params = line value
        return np.stack([param_values] * len(param_cols), axis=1)
    return lines_to_params_callabe


@pytest.fixture()
def mappings_grid(monkeypatch, grid_df, line_cols, param_cols, lines_to_params, params_to_lines):
    def mock_fit(self):
        self.lines_to_params = lines_to_params
        self.params_to_lines = params_to_lines
    monkeypatch.setattr(backward_forward_model.MappingsGrid, 'fit', mock_fit)
    return backward_forward_model.MappingsGrid(grid_df, line_cols, param_cols)


@pytest.fixture()
def diagnostic_model_unfit(grid_df, galaxies, line_cols, param_cols, mappings_grid):
    return backward_forward_model.DiagnosticModel(grid_df, galaxies, line_cols, param_cols, mappings_grid)


@pytest.fixture()
def diagnostic_model_fit(
        diagnostic_model_unfit,
        galaxies_with_params,
        lines_to_params,
        params_to_lines,
        param_prior):
    model = copy(diagnostic_model_unfit)
    model.fitted = True
    model.galaxies = galaxies_with_params
    model.lines_to_params = lines_to_params
    model.params_to_lines = params_to_lines
    model.param_prior = param_prior
    return model


@pytest.fixture()
def grid_df(line_cols, param_cols, params_to_lines):  # implicit required match with param cols
    """Emission lines and parameters (e.g. ionization) from simulated MAPPINGS grid
    """
    n_grid_points = 50
    param_values = np.zeros((n_grid_points, len(param_cols)))
    for n in range(len(param_cols)):  # could surely do cleaner than this
        param_values[:, n] = np.linspace(0., 1.)

    # actually calling the inner callable
    line_values = params_to_lines(param_values)

    data = np.concatenate([param_values, line_values], axis=1)
    return pd.DataFrame(
        data=data,
        columns=param_cols + line_cols)


@pytest.fixture()
def param_prior(param_cols):  # implicit required match with param cols
    # input (n_samples, n_features), output ones like (n_samples)
    return lambda x: np.ones(len(x)) * 0.5  # always return .5


@pytest.fixture()
def galaxies(line_cols):
    return pd.DataFrame(
        data=np.random.uniform(size=(20, len(line_cols))),
        columns=line_cols)


@pytest.fixture()
def galaxies_with_params(grid_df):
    return grid_df.sample(10)  # pretend the galaxies are exactly on the grid


def test_grid_to_bpt_data(grid_loc):
    df = backward_forward_model.grid_to_bpt_data(grid_loc)
    # agn_df = grid_to_bpt_data(agn_nlr_grid_loc)


def test_fit_lines_to_params(grid_df, line_cols, param_cols):
    lines_to_params = backward_forward_model.fit_lines_to_params(
        grid_df, line_cols, param_cols)


def test_fit_params_to_lines(grid_df, line_cols, param_cols):
    params_to_lines = backward_forward_model.fit_params_to_lines(
        grid_df, line_cols, param_cols)


def test_estimate_params_of_galaxies(galaxies, lines_to_params, line_cols, param_cols):
    galaxies_with_params = backward_forward_model.estimate_params_of_galaxies(
        galaxies, lines_to_params, line_cols, param_cols)


def test_estimate_param_prior(grid_df, param_cols):
    prior = backward_forward_model.estimate_param_prior(grid_df, param_cols)
    assert callable(prior)


def test_diagnostic_model_fit(diagnostic_model_unfit):
    diagnostic_model_unfit.fit()
    assert diagnostic_model_unfit.fitted


def test_diagnostic_model_forward_pass(diagnostic_model_fit, line_cols):
    synthetic_df = diagnostic_model_fit.forward_pass(n_samples=10)
    fig, ax = plt.subplots()
    # bpt_utilities.show_bpt_diagram_static(synthetic_df, ax)
    ax.scatter(synthetic_df[line_cols[0]], synthetic_df[line_cols[1]], alpha=0.8)
    ax.set_xlabel(line_cols[0])
    ax.set_ylabel(line_cols[1])
    ax.set_title('Synthetic Galaxies')
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'diagnostic_forward_pass.png'))


def test_diagnostic_model_describe(tmpdir, diagnostic_model_fit):
    diagnostic_model_fit.describe(os.path.join(TEST_FIGURE_DIR, 'describe_diagnostic_model'))
