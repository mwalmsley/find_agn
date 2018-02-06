
from astropy.table import Table
from astropy.io import fits
import pandas as pd
import numpy as np
import datetime


def cache_nsa(nsa_loc, cache_loc, useful_cols):
    print('Begin caching at {}'.format(current_time()))
    nsa_loc = '/data/galaxy_zoo/decals/catalogs/nsa_v1_0_1.fits'
    nsa = Table(fits.getdata(nsa_loc))
    # split ELPETRO_FLUX
    print('Table loaded at {}'.format(current_time()))
    sdss_bands = ['f', 'n', 'u', 'g', 'r', 'i', 'z']
    elpetro_columns = ['elpetro_flux_{}'.format(band) for band in sdss_bands]
    for n in range(len(elpetro_columns)):
        col = elpetro_columns[n]
        nsa[col] = [nsa[row]['ELPETRO_FLUX'][n] for row in range(len(nsa))]
    useful_cols = useful_cols + elpetro_columns
    print(useful_cols)
    print(nsa[useful_cols][0])
    nsa = nsa[useful_cols].to_pandas()
    new_names = [col.lower() for col in useful_cols]
    renaming_dict = dict(zip(useful_cols, new_names))
    nsa = nsa.rename(columns=renaming_dict)
    print('Saving Table at {}'.format(current_time()))
    nsa.to_csv(cache_loc, index=False)
    print('Saved to csv at {}'.format(current_time()))
    return nsa


def get_agn_catalog(agn_catalog_loc):
    catalog = Table.read(agn_catalog_loc).to_pandas()
    # nothing else to do for now
    return catalog


def current_time():
    return datetime.datetime.now().time()

if __name__ == '__main__':
    useful_nsa_columns = [
        'IAUNAME',
        'RA',
        'DEC',
        # 'BASTOKES',
        # 'PHISTOKES',
        'PETRO_BA50',
        'PETRO_PHI50',
        'PETRO_BA90',
        'PETRO_PHI90'
    ]
    nsa_loc = '/data/galaxy_zoo/decals/catalogs/nsa_v1_0_1.fits'
    nsa_cache_loc = '/data/galaxy_zoo/decals/catalogs/nsa_v1`_0_1_cached.csv'
    nsa = cache_nsa(nsa_loc, nsa_cache_loc, useful_nsa_columns)
