import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import table


def match_galaxies_to_catalog(galaxies, catalog, matching_radius=10 * u.arcsec,
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