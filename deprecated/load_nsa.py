
import datetime
from astropy.io import fits
from astropy.table import Table

print(datetime.datetime.now().time())
nsa_loc = '/data/galaxy_zoo/decals/catalogs/nsa_v1_0_1.fits'
# catalog = Table(fits.getdata(nsa_loc))
catalog = Table.read(nsa_loc)
print(len(catalog))
print(datetime.datetime.now().time())