from functools import reduce
import numpy as np
import iarray as ia
import dask
import dask.array as da
import zarr

precip_mean_disk = ia.load("mean-3m.iarr")
blocksize = reduce(lambda x, y: x * y, precip_mean_disk.blockshape) * np.dtype(np.float32).itemsize
shape = precip_mean_disk.shape
chunks = precip_mean_disk.chunkshape
dtype = np.float32
clevel = ia.Config().clevel

precip = zarr.open("precip-3m.zarr")
d = da.from_zarr(precip)
precip1 = d[0]
precip2 = d[1]
precip3 = d[2]

@profile
def dask_mean_disk(expr):
    with dask.config.set(scheduler="threads"):
        expr_val = zarr.open(
            "mean-3m.zarr",
            "w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=zarr.Blosc(clevel=clevel, cname="lz4", blocksize=blocksize),
        )
        da.to_zarr(expr, expr_val)
    return expr_val

mean_expr = (precip1 + precip2 + precip3) / 3
mean_disk = dask_mean_disk(mean_expr)

@profile
def dask_trans_disk(expr):
    with dask.config.set(scheduler="threads"):
        expr_val = zarr.open(
            "trans-3m.zarr",
            "w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=zarr.Blosc(clevel=clevel, cname="lz4", blocksize=blocksize),
        )
        da.to_zarr(expr, expr_val)
    return expr_val

trans_expr = np.tan(precip1) * (np.sin(precip1) * np.sin(precip2) + np.cos(precip2)) + np.sqrt(precip3) * 2
trans_disk = dask_trans_disk(trans_expr)
