import numpy as np
from numcodecs import Blosc
import zarr
import dask.array as da
import dask
import sys

import ctypes

mkl_rt = ctypes.CDLL("libmkl_rt.dylib")
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_set_num_threads(1)

persistent = True

shape = (8192, 8192)
chunkshape = (2048, 2048)

if len(sys.argv) == 1:
    nthreads = 1
else:
    nthreads = int(sys.argv[1])

cname = "lz4"
clevel = 5
shuffle = Blosc.SHUFFLE

if persistent:
    import os
    import shutil

    afilename = "a.zarr"
    if os.path.exists(afilename):
        shutil.rmtree(afilename)
    bfilename = "b.zarr"
    if os.path.exists(bfilename):
        shutil.rmtree(bfilename)
    cfilename = "c.zarr"
    if os.path.exists(cfilename):
        shutil.rmtree(cfilename)

compressor = Blosc(cname=cname, clevel=clevel, shuffle=shuffle)


@profile
def create_data():
    if persistent:
        azarr = zarr.open(
            afilename,
            mode="w",
            shape=shape,
            chunks=chunkshape,
            dtype=np.float64,
            compressor=compressor,
        )
        bzarr = zarr.open(
            bfilename,
            mode="w",
            shape=shape,
            chunks=chunkshape,
            dtype=np.float64,
            compressor=compressor,
        )
    else:
        azarr = zarr.empty(shape=shape, chunks=chunkshape, dtype=np.float64, compressor=compressor)
        bzarr = zarr.empty(shape=shape, chunks=chunkshape, dtype=np.float64, compressor=compressor)

    tmp = np.linspace(-1, 1, int(np.prod(shape))).reshape(shape)
    azarr[:] = tmp
    bzarr[:] = tmp
    del tmp
    return azarr, bzarr


azarr, bzarr = create_data()


@profile
def matmul():
    scheduler = "single-threaded" if nthreads == 1 else "threads"
    with dask.config.set(scheduler=scheduler, num_workers=nthreads):
        if persistent:
            czarr = zarr.open(
                cfilename,
                mode="w",
                shape=shape,
                chunks=chunkshape,
                dtype=np.float64,
                compressor=compressor,
            )
        else:
            czarr = zarr.empty(
                shape=shape, chunks=chunkshape, dtype=np.float64, compressor=compressor
            )

        adask = da.from_zarr(azarr)
        bdask = da.from_zarr(bzarr)
        cdask = da.matmul(adask, bdask)
        da.to_zarr(cdask, czarr)
        return czarr


czarr = matmul()

print(azarr.info)
print(bzarr.info)
print(czarr.info)

if persistent:
    if os.path.exists(afilename):
        shutil.rmtree(afilename)
    if os.path.exists(bfilename):
        shutil.rmtree(bfilename)
    if os.path.exists(cfilename):
        shutil.rmtree(cfilename)
