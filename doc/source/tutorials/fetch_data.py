import os
import sys
import xarray as xr
import numpy as np
import s3fs
import iarray as ia
import zarr
from numcodecs import Blosc


def open_zarr(year, month, datestart, dateend):
    fs = s3fs.S3FileSystem(anon=True)
    datestring = 'era5-pds/zarr/{year}/{month:02d}/data/'.format(year=year, month=month)
    s3map = s3fs.S3Map(datestring + 'precipitation_amount_1hour_Accumulation.zarr/', s3=fs)
    precip_zarr = xr.open_dataset(s3map, engine="zarr")
    precip_zarr = precip_zarr.sel(time1=slice(np.datetime64(datestart), np.datetime64(dateend)))

    return precip_zarr.precipitation_amount_1hour_Accumulation

# WARNING: this is for debugging purposes only. In production comment out the line below!
# if os.path.exists("precip-3m.iarr"): os.remove("precip-3m.iarr")
if os.path.exists("precip-3m.iarr"):
    print("Dataset %s is already here!" % "precip-3m.iarr")
    sys.exit(0)

print("Fetching data from S3 (era5-pds)...")
precip_m0 = open_zarr(1987, 10, '1987-10-01', '1987-10-30 23:59')
precip_m1 = open_zarr(1987, 11, '1987-11-01', '1987-11-30 23:59')
precip_m2 = open_zarr(1987, 12, '1987-12-01', '1987-12-30 23:59')

for path in ("precip1.iarr", "precip2.iarr", "precip3.iarr"):
    if os.path.exists(path):
        os.remove(path)

ia.set_config(favor=ia.Favors.CRATIO)
m_shape = precip_m0.shape
ia_precip0 = ia.empty(m_shape, dtype=np.float32, urlpath="precip1.iarr")
ia_precip1 = ia.empty(m_shape, dtype=np.float32, urlpath="precip2.iarr")
ia_precip2 = ia.empty(m_shape, dtype=np.float32, urlpath="precip3.iarr")
ia_precip = ia.empty((3, ) + m_shape, dtype=np.float32, urlpath="precip-3m.iarr")

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
za_precip0 = zarr.open('precip1.zarr', mode='w', shape=m_shape, dtype=np.float32, compressor=compressor)
za_precip1 = zarr.open('precip2.zarr', mode='w', shape=m_shape, dtype=np.float32, compressor=compressor)
za_precip2 = zarr.open('precip3.zarr', mode='w', shape=m_shape, dtype=np.float32, compressor=compressor)
za_precip = zarr.open('precip-3m.zarr', mode='w', shape=(3,) + m_shape, dtype=np.float32, compressor=compressor)

print("Fetching and storing 1st month...")
values = precip_m0.values
ia_precip0[:] = values
ia_precip[0] = values
za_precip0[:] = values
za_precip[0] = values

print("Fetching and storing 2nd month...")
values = precip_m1.values
ia_precip1[:] = values
ia_precip[1] = values
za_precip1[:] = values
za_precip[1] = values

print("Fetching and storing 3rd month...")
values = precip_m2.values
ia_precip2[:] = values
ia_precip[2] = values
za_precip2[:] = values
za_precip[2] = values
