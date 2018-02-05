from astropy.io import fits
import pandas as pd
from astropy.table import Table
from shared_utilities import match_galaxies_to_catalog_pandas
import numpy as np

import matplotlib.pyplot as plt

def get_heckman_catalog(heckman_loc):
    df = pd.read_table(heckman_loc, sep=r"\s*")
    old_columns = df.columns.values
    new_columns = map(lambda x: x.lower().strip('#'), old_columns)
    df = df.rename(columns=dict(zip(old_columns, new_columns)))
    df = df.rename(
        columns={'main_samp': 'main_sample'})
    assert df[(df['herg'] == 1) & (df['lerg']==1)].empty
    return df


joint_catalog = Table(fits.getdata('/Volumes/external/decals/catalogs/dr5_nsa1_0_0_to_upload.fits'))
print(len(joint_catalog))
heckman = get_heckman_catalog(heckman_loc='sdss_dr7_radiosources.cat')
print(len(heckman))

matched, _ = match_galaxies_to_catalog_pandas(heckman, joint_catalog[['ra', 'dec']].to_pandas())
print(len(matched))
print(matched)

joint_catalog = joint_catalog[joint_catalog[np.random.choice([True, False, False, False, False, False, False], len(joint_catalog))]]

plt.scatter(np.array(heckman['ra']), np.array(heckman['dec']))
plt.scatter(np.array(joint_catalog['ra']), np.array(joint_catalog['dec']), c='r')
plt.show()
