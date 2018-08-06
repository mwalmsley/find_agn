import pytest


def test_template_calculate(ref_template, model_dir):
    # make a model template
    assert not ref_template.calculated
    ref_template.calculate(model_dir)
    assert ref_template.calculated


def test_template_save(calculated_template, empty_db, save_dir):
    calculated_template.save(empty_db, save_dir)
    cursor = empty_db.cursor()  # hopefully no longer empty
    cursor.execute('''SELECT id, model_loc FROM dual_burst_models''')
    model = cursor.fetchone()
    assert model[0] > 1533543091450000  # id is current timestamp
    assert model[1][-6:] == '.model'  # model_loc is location of saved model


def test_template_load(ref_template, filled_db):
    # note: depends on previously saved example model in test_examples
    ref_template.modelset = 'bc03'
    ref_template.metallicity = 0.05
    ref_template.imf = 'salp'
    # etc for star history
    ref_template.load(filled_db)  # includes entry with these params
    assert ref_template.calculated
    sed = ref_template.model.get_sed_z(zf=2., z=0.05)


def test_template_get_continuum(loaded_template):
    continuum = loaded_template.get_continuum(z=0.05)
    assert continuum.line_data is None  # haven't calculated yet
    assert continuum.background is None

# # To minimise saved model grid dimensions,
# # make all galaxies the same age: 8 Gyr from formation
# current_galaxy_age = 8 * units.Gyr
# current_z = 0.05  # defined to be here, very small error introduced
# formation_z = notebook_utils.formation_redshift_for_fixed_age(current_z, current_galaxy_age)
