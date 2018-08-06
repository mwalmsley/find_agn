import sqlite3
import copy

import pytest
import numpy as np
import scipy

from find_agn import star_formation, templates, spectral_fitting, continuums, tests


@pytest.fixture()
def model_dir():
    return tests.TEST_EXAMPLE_DIR


@pytest.fixture()
def save_dir(tmpdir):
    return tmpdir.mkdir('save_dir').strpath


@pytest.fixture()
def star_history():
    """
    DualBurstHistory: standard dual burst star formation history object
    """
    current_z = 0.1  # let's just assume and ignore for now
    formation_z = 1.5  # let's just assume and ignore for now
    first_duration = 1.  # Gyr
    second_duration = .5  # Gyr
    return star_formation.DualBurstHistory(
        current_z=current_z,
        formation_z=formation_z,
        first_duration=first_duration,
        second_duration=second_duration)


@pytest.fixture()
def empty_db():
    """
    sqlite database with dual_burst_models table
    """
    db = sqlite3.connect(':memory:')
    cursor = db.cursor()
    cursor.execute('''
                CREATE TABLE dual_burst_models(
                    id INTEGER PRIMARY KEY,
                    model_loc TEXT,
                    modelset TEXT,
                    metallicity FLOAT,
                    imf FLOAT,
                    current_z FLOAT,
                    formation_z FLOAT, 
                    first_duration FLOAT, 
                    second_duration FLOAT)
                        ''')
    db.commit()
    return db


@pytest.fixture()
def filled_db(empty_db):
    cursor = empty_db.cursor()
    cursor.execute(
        '''
        INSERT INTO dual_burst_models(
            id,
            model_loc,
            modelset,
            metallicity,
            imf,
            current_z,
            formation_z,
            first_duration,
            second_duration
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            1533558377173919,
            'find_agn/tests/test_examples/1533558377173919.model',
            'bc03',
            0.05,
            'salp',
            0.1,
            1.5,
            1.,
            0.5
        )
    )
    empty_db.commit()
    return empty_db  # now has an entry


@pytest.fixture()
def ref_template(star_history):
    return templates.Template(star_history=star_history)


@pytest.fixture()
def calculated_template(ref_template, calculated_ezgal_model):
    calc_template = copy.copy(ref_template)
    calc_template.calculated = True
    calc_template.model = calculated_ezgal_model
    return calc_template


@pytest.fixture
def calculated_ezgal_model():
    return MockCalculatedModel()


class MockCalculatedModel():  # mock EzGal model to save
    def save_model(self, save_loc, filter_info):
        with open(save_loc, 'w') as f:
            f.write('Hello, I am a saved model')


@pytest.fixture
def loaded_template(filled_db, ref_template):
    ref_template.modelset = 'bc03'
    ref_template.metallicity = 0.05
    ref_template.imf = 'salp'
    ref_template.load(filled_db)  # includes entry with these params
    assert ref_template.calculated
    return ref_template  # now a loaded template


@pytest.fixture()
def ref_continuum_raw():
    wavelength = np.linspace(1000, 10000, num=10000)
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

    freq = 299792458. / wavelength
    return continuums.Continuum(freq, flux_density)


@pytest.fixture()
def ref_continuum_prepared():
    wavelength = np.linspace(4500, 7000, num=10000)  # narrower range
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
    freq = 299792458. / wavelength
    return continuums.Continuum(freq, flux_density, background=continuum)  # background known

    # plt.clf()
    # plt.plot(wavelength, flux_density)
    # plt.savefig(TEST_FIGURE_DIR + '/reference_spectrum_prepared.png')
