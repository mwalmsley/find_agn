
from astropy.table import Table
import pandas as pd
import numpy as np


def get_agn_catalog(agn_catalog_loc):
    catalog = Table.read(agn_catalog_loc).to_pandas()
    # nothing else to do for now
    return catalog

def get_nsa(nsa_loc, useful_cols, cache_loc=None):
    # nsa = Table.read(nsa_loc, format='fits')
    # nsa = nsa[useful_cols].to_pandas()
    # nsa = pd.read_csv(nsa_loc, usecols=useful_cols, encoding='latin1')
    nsa = pd.read_csv(nsa_loc, nrows=10000, encoding='latin1')
    # if cache_loc is not None:
    #     nsa.to_csv(cache_loc)
    # nsa = nsa.rename([{col: col.lower()} for col in nsa.columns.values])
    return nsa

# nsa = get_nsa(nsa_loc)
# print(catalog.head())
# axial_hist = hv.Histogram(np.histogram(catalog['ra'], bins=24))

