import pytest

from find_agn import templates, get_template_grid

@pytest.fixture()
def metallicity_range():
    return [0.008, 0.02, 0.05, 1.02]  # must match model metallicities in test_examples dir


def test_make_metallicity_grid(
    metallicity_range,
    star_history,
    model_dir,
    empty_db,
    save_dir,
    calculated_template,
    calculated_ezgal_model,
    monkeypatch
):

    # for speed, don't actually run the ezgal calculations
    def mock_calculate(self, model_dir):
        self.calculated = True
        self.model = calculated_ezgal_model
    monkeypatch.setattr(templates.Template, 'calculate', mock_calculate)

    # for fixed model assumptions, look at one model over many ages/history snapshots
    get_template_grid.make_metallicity_grid(
        metallicity_range,
        star_history,
        empty_db,
        model_dir,
        save_dir
    )

    cursor = empty_db.cursor()  # no longer empty, hopefully

    cursor.execute('''SELECT id, model_loc FROM dual_burst_models''')
    models = list(cursor)  # unpack search iterable for convenience
    assert len(models) == len(metallicity_range)
    for model in models:
        assert model[0] > 1533548684743653
        assert model[1][-6:] == '.model'
