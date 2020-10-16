import numpy as np
from numcodecs import Blosc
import zarr
import dask.array as da
import dask
import sys
import os
import shutil
import ctypes

mkl_rt = ctypes.CDLL("libmkl_rt.so")
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_set_num_threads(1)

persistent = True

shape = (16384, 16384)
chunkshape = (4096, 4096)

shape = (1000, 1000)
chunkshape = (500, 500)

if len(sys.argv) == 1:
    nthreads = 1
else:
    nthreads = int(sys.argv[1])

cname = "lz4"
clevel = 5
shuffle = Blosc.SHUFFLE

afilename = "a.zarr"
bfilename = "b.zarr"
cfilename = "c.zarr"

compressor = Blosc(cname=cname, clevel=clevel, shuffle=shuffle)

if persistent:
    if not os.path.exists(afilename):
        azarr = zarr.open(
            afilename,
            mode="w",
            shape=shape,
            chunks=chunkshape,
            dtype=np.float64,
            compressor=compressor,
        )
        tmp = np.linspace(-1, 1, int(np.prod(shape))).reshape(shape)
        azarr[:] = tmp
    else:
        azarr = zarr.load("a.zarr")
    if not os.path.exists(bfilename):
        bzarr = zarr.open(
            bfilename,
            mode="w",
            shape=shape,
            chunks=chunkshape,
            dtype=np.float64,
            compressor=compressor,
        )
        tmp = np.linspace(-1, 1, int(np.prod(shape))).reshape(shape)
        bzarr[:] = tmp
    else:
        bzarr = zarr.load("b.zarr")
else:
    azarr = zarr.empty(shape=shape, chunks=chunkshape, dtype=np.float64, compressor=compressor)
    bzarr = zarr.empty(shape=shape, chunks=chunkshape, dtype=np.float64, compressor=compressor)


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
        czarr = zarr.empty(shape=shape, chunks=chunkshape, dtype=np.float64, compressor=compressor)

    adask = da.from_zarr(azarr)
    bdask = da.from_zarr(bzarr)
    cdask = da.matmul(adask, bdask)
    da.to_zarr(cdask, czarr)


if persistent:
    # if os.path.exists(afilename):
    #     shutil.rmtree(afilename)
    # if os.path.exists(bfilename):
    #     shutil.rmtree(bfilename)
    if os.path.exists(cfilename):
        shutil.rmtree(cfilename)
