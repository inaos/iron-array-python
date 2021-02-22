import numpy as np
from numcodecs import Blosc
import zarr
import dask.array as da
import dask
import sys
import os
import shutil
import ctypes
from time import time

mkl_rt = ctypes.CDLL("libmkl_rt.so")
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_set_num_threads(1)

persistent = True

shape = (16384, 16384)
chunkshape = (4096, 4096)

# shape = (1000, 1000)
# chunkshape = (500, 500)

if len(sys.argv) == 1:
    nthreads = 8
else:
    nthreads = int(sys.argv[1])

cname = "lz4"
clevel = 5
shuffle = Blosc.SHUFFLE

aurlpath = "a.zarr"
burlpath = "b.zarr"
curlpath = "c.zarr"

compressor = Blosc(cname=cname, clevel=clevel, shuffle=shuffle)

if persistent:
    print("(Re-)Generating operand A")
    if not os.path.exists(aurlpath):
        azarr = zarr.open(
            aurlpath,
            mode="w",
            shape=shape,
            chunks=chunkshape,
            dtype=np.float64,
            compressor=compressor,
        )
        tmp = np.linspace(-1, 1, int(np.prod(shape))).reshape(shape)
        azarr[:] = tmp
    else:
        azarr = zarr.open("a.zarr")
        if azarr.shape != shape or azarr.chunks != chunkshape:
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            if os.path.exists(aurlpath):
                shutil.rmtree(aurlpath)
            azarr = zarr.open(
                aurlpath,
                mode="w",
                shape=shape,
                chunks=chunkshape,
                dtype=np.float64,
                compressor=compressor,
            )
            tmp = np.linspace(-1, 1, int(np.prod(shape))).reshape(shape)
            azarr[:] = tmp

    print("(Re-)Generating operand B")
    if not os.path.exists(burlpath):
        bzarr = zarr.open(
            burlpath,
            mode="w",
            shape=shape,
            chunks=chunkshape,
            dtype=np.float64,
            compressor=compressor,
        )
        tmp = np.linspace(-1, 1, int(np.prod(shape))).reshape(shape)
        bzarr[:] = tmp
    else:
        print("(Re-)Generating operand B")
        bzarr = zarr.open("b.zarr")
        if bzarr.shape != shape or bzarr.chunks != chunkshape:
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            if os.path.exists(burlpath):
                shutil.rmtree(burlpath)
            bzarr = zarr.open(
                burlpath,
                mode="w",
                shape=shape,
                chunks=chunkshape,
                dtype=np.float64,
                compressor=compressor,
            )
            tmp = np.linspace(-1, 1, int(np.prod(shape))).reshape(shape)
            bzarr[:] = tmp
else:
    azarr = zarr.empty(shape=shape, chunks=chunkshape, dtype=np.float64, compressor=compressor)
    bzarr = zarr.empty(shape=shape, chunks=chunkshape, dtype=np.float64, compressor=compressor)

if persistent:
    if os.path.exists(curlpath):
        shutil.rmtree(curlpath)

scheduler = "single-threaded" if nthreads == 1 else "threads"
with dask.config.set(scheduler=scheduler, num_workers=nthreads):
    if persistent:
        czarr = zarr.open(
            curlpath,
            mode="w",
            shape=shape,
            chunks=chunkshape,
            dtype=np.float64,
            compressor=compressor,
        )
    else:
        czarr = zarr.empty(shape=shape, chunks=chunkshape, dtype=np.float64, compressor=compressor)

    print(f"Start actual matmul with nthreads = {nthreads}")
    t0 = time()
    adask = da.from_zarr(azarr)
    bdask = da.from_zarr(bzarr)
    cdask = da.matmul(adask, bdask)
    da.to_zarr(cdask, czarr)
    print("Time for iarray matmul:", round((time() - t0), 3))


if persistent:
    # if os.path.exists(aurlpath):
    #     shutil.rmtree(aurlpath)
    # if os.path.exists(burlpath):
    #     shutil.rmtree(burlpath)
    if os.path.exists(curlpath):
        shutil.rmtree(curlpath)
