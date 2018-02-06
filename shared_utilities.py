import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import table


import pandas as pd

from astropy.coordinates import SkyCoord
from astropy import units as u

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image


def plot_catalog(catalog, filename, column_to_colour=None):
    """

    Args:
        catalog ():
        filename ():
        column_to_colour ():

    Returns:

    """
    canvas = ds.Canvas(plot_width=300, plot_height=300)
    if column_to_colour:  # shade pixels by average value in column_to_colour
        aggc = canvas.points(catalog, 'ra', 'dec', ds.mean(column_to_colour))
    else:
        aggc = canvas.points(catalog, 'ra', 'dec')
    img = tf.shade(aggc)
    export_image(img, filename)


def plot_catalog_overlap(catalog_a, catalog_b, legend, filename):
    """

    Args:
        catalog_a ():
        catalog_b ():
        legend ():
        filename ():

    Returns:

    """

    a_coords = catalog_a[['ra', 'dec']]
    a_coords['catalog'] = legend[0]
    b_coords = catalog_b[['ra', 'dec']]
    b_coords['catalog'] = legend[1]

    df_to_plot = pd.concat([a_coords, b_coords])
    df_to_plot['catalog'] = df_to_plot['catalog'].astype('category')

    canvas = ds.Canvas(plot_width=300, plot_height=300)
    aggc = canvas.points(df_to_plot, 'ra', 'dec', ds.count_cat('catalog'))
    img = tf.shade(aggc)
    export_image(img, filename)


def match_galaxies_to_catalog_table(galaxies, catalog, matching_radius=10 * u.arcsec,
                              galaxy_suffix='_subject', catalog_suffix=''):

    galaxies_coord = SkyCoord(ra=galaxies['ra'] * u.degree, dec=galaxies['dec'] * u.degree)
    catalog_coord = SkyCoord(ra=catalog['ra'] * u.degree, dec=catalog['dec'] * u.degree)

    catalog['best_match'] = np.arange(len(catalog))
    best_match_catalog_index, sky_separation, _ = galaxies_coord.match_to_catalog_sky(catalog_coord)
    galaxies['best_match'] = best_match_catalog_index
    galaxies['sky_separation'] = sky_separation.to(u.arcsec).value
    matched_galaxies = galaxies[galaxies['sky_separation'] < matching_radius.value]

    matched_catalog = table.join(matched_galaxies,
                                 catalog,
                                 keys='best_match',
                                 join_type='inner',
                                 table_names=['{}'.format(galaxy_suffix), '{}'.format(catalog_suffix)],
                                 uniq_col_name='{col_name}{table_name}')
    # correct names not shared
    unmatched_galaxies = galaxies[galaxies['sky_separation'] >= matching_radius.value]
    return matched_catalog, unmatched_galaxies


def match_galaxies_to_catalog_pandas(galaxies, catalog, matching_radius=10 * u.arcsec,
                              galaxy_suffix='_subject', catalog_suffix=''):

    galaxies_coord = SkyCoord(ra=galaxies['ra'].values * u.degree, dec=galaxies['dec'].values * u.degree)
    catalog_coord = SkyCoord(ra=catalog['ra'].values * u.degree, dec=catalog['dec'].values * u.degree)

    catalog['best_match'] = np.arange(len(catalog))
    best_match_catalog_index, sky_separation, _ = galaxies_coord.match_to_catalog_sky(catalog_coord)
    galaxies['best_match'] = best_match_catalog_index
    galaxies['sky_separation'] = sky_separation.to(u.arcsec).value
    matched_galaxies = galaxies[galaxies['sky_separation'] < matching_radius.value]

    matched_catalog = pd.merge(
        matched_galaxies,
        catalog,
        on='best_match',
        how='inner',
        suffixes=['{}'.format(galaxy_suffix), '{}'.format(catalog_suffix)]
    )
    # correct names not shared
    unmatched_galaxies = galaxies[galaxies['sky_separation'] >= matching_radius.value]
    return matched_catalog, unmatched_galaxies