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
    assert df[(df['herg'] == 1) & (df['lerg'] == 1)].empty
    # convert ra from hours to degrees
    df['ra'] = df['ra'] * 15  # i.e. / 24, * 365
    return df


def plot_overlaps(joint_catalog, heckman):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.hist(heckman['ra'])
    ax1.set_xlabel('heckman ra (h)')
    ax2.hist(heckman['dec'])
    ax2.set_xlabel('heckman dec (deg)')
    ax3.hist(joint_catalog['ra'])
    ax3.set_xlabel('decals ra (deg)')
    ax4.hist(joint_catalog['dec'])
    ax4.set_xlabel('decals dec (deg)')
    fig.tight_layout()
    fig.savefig('heckman_joint_dist.png')
    plt.clf()

    sample_index = np.array(np.around(0.6 * np.random.rand(len(joint_catalog))).astype(bool))
    sampled_joint_catalog = joint_catalog[sample_index]
    fig, ax = plt.subplots()
    ax.scatter(sampled_joint_catalog['ra'], sampled_joint_catalog['dec'], c='k', alpha=0.3, s=0.2)
    ax.scatter(heckman['ra'], heckman['dec'], s=0.2)
    ax.set_xlabel('ra (deg)')
    ax.set_ylabel('dec (deg)')
    ax.legend(['DECALS in SDSS', 'Heckman et al 2012'], loc=0)
    fig.tight_layout()
    fig.savefig('heckman_v_joint.png')
    plt.clf()
