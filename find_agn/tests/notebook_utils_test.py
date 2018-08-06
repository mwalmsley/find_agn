import pytest

from astropy import units

from find_agn import notebook_utils


def test_formation_refshift_for_fixed_age():
    formation_z = notebook_utils.formation_redshift_for_fixed_age(0.05, 8 * units.Gyr)
    assert formation_z > 1.
