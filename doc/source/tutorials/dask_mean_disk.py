from functools import reduce
import numpy as np
import iarray as ia
import dask
import dask.array as da
import zarr

# Use a previous computation for getting metadata (shape, chunks...)
precip = zarr.open("precip-3m.zarr")
shape = precip.shape[1:]
dtype = np.float32
clevel = ia.Config().clevel
compressor = zarr.Blosc(clevel=clevel, cname="lz4")

#zarr.save("precip1.zarr", precip[0])
#zarr.save("precip2.zarr", precip[1])
#zarr.save("precip3.zarr", precip[2])

zprecip1 = zarr.open("precip1.zarr")
zprecip2 = zarr.open("precip2.zarr")
zprecip3 = zarr.open("precip3.zarr")

precip1 = da.from_zarr(zprecip1)
precip2 = da.from_zarr(zprecip2)
precip3 = da.from_zarr(zprecip3)


@profile
def dask_mean_disk(expr):
    with dask.config.set(scheduler="threads", num_workers=None):
        expr_val = zarr.open(
            "mean-3m.zarr",
            "w",
            shape=shape,
            dtype=dtype,
            compressor=compressor,
        )
        da.to_zarr(expr, expr_val)
    return expr_val

mean_expr = (precip1 + precip2 + precip3) / 3
mean_disk = dask_mean_disk(mean_expr)

@profile
def dask_trans_disk(expr):
    with dask.config.set(scheduler="threads", num_workers=None):
        expr_val = zarr.open(
            "trans-3m.zarr",
            "w",
            shape=shape,
            dtype=dtype,
            compressor=compressor,
        )
        da.to_zarr(expr, expr_val)
    return expr_val

trans_expr = np.tan(precip1) * (np.sin(precip1) * np.sin(precip2) + np.cos(precip2)) + np.sqrt(precip3) * 2
trans_disk = dask_trans_disk(trans_expr)
