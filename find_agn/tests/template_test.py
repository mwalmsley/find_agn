import pytest

from astropy import units

from find_agn import notebook_utils, star_formation, template


@pytest.fixture()
def star_history():
        first_duration = 1.  # Gyr
        second_duration = .5  # Gyr
        return star_formation.DualBurstHistory(
            current_z=current_z,
            formation_z=formation_z,
            first_duration=first_duration,
            second_duration=second_duration)


@pytest.fixture()
def ref_template(star_history):
        return template.Template()



# To minimise saved model grid dimensions, 
# make all galaxies the same age: 8 Gyr from formation
current_galaxy_age = 8 * units.Gyr
current_z = 0.05  # defined to be here, very small error introduced
formation_z = notebook_utils.formation_redshift_for_fixed_age(current_z, current_galaxy_age)

def test_template_calculate(ref_template):
        # make a model template
        ref_template.calculate()
