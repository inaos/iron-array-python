from time import time
from functools import reduce
import ctypes
import dask
import dask.array as da
from numcodecs import Blosc
import numpy as np
import zarr

import iarray as ia

mkl_rt = ctypes.CDLL("libmkl_rt.dylib")
mkl_get_max_threads = mkl_rt.mkl_get_max_threads


def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))


MEMPROF = True
if MEMPROF:
    from memory_profiler import profile
else:

    def profile(f):
        return f


DTYPE = np.float32
NTHREADS = 8
CLEVEL = 5
CODEC = ia.Codecs.LZ4

t_iarray = []
t_dask = []
t_ratio = []

ashape = (10000, 10000)
achunkshape = (500, 500)
ablockshape = (128, 128)

bshape = (10000, 8000)
bchunkshape = (250, 500)
bblockshape = (128, 128)

cchunkshape = (500, 500)
cblockshape = (128, 128)


compressor = Blosc(
    cname="lz4",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    blocksize=reduce(lambda x, y: x * y, ablockshape),
)
ia.set_config(codec=CODEC, clevel=CLEVEL, nthreads=NTHREADS)

astorage = ia.Storage(achunkshape, ablockshape)
dtshape = ia.DTShape(ashape, dtype=DTYPE)
lia = ia.linspace(dtshape, 0, 1, storage=astorage)
nia = ia.random_normal(
    dtshape,
    0,
    0.0000001,
    storage=astorage,
)
aia = (lia + nia).eval(dtshape, storage=astorage)

bstorage = ia.Storage(bchunkshape, bblockshape)
dtshape = ia.DTShape(bshape, dtype=DTYPE)
lia = ia.linspace(dtshape, 0, 1, storage=bstorage)
nia = ia.random_normal(dtshape, 0, 0.0000001, storage=bstorage)
bia = (lia + nia).eval(dtshape, storage=bstorage)

ablock = (500, 500)
bblock = (500, 500)
cstorage = ia.Storage(cchunkshape, cblockshape)


@profile
def ia_matmul(aia, bia, ablock, bblock):
    return ia.matmul(aia, bia, ablock, bblock, storage=cstorage)


t0 = time()
cia = ia_matmul(aia, bia, ablock, bblock)
t1 = time()
tia = t1 - t0
print("Time for computing matmul (via iarray): %.3f" % (tia))
print(f"a cratio: {aia.cratio}")
print(f"b cratio: {bia.cratio}")
print(f"out cratio: {cia.cratio}")

azarr = zarr.empty(shape=ashape, chunks=achunkshape, dtype=DTYPE, compressor=compressor)
for info, block in aia.iter_read_block(achunkshape):
    sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
    azarr[sl] = block[:]


bzarr = zarr.empty(shape=bshape, chunks=bchunkshape, dtype=DTYPE, compressor=compressor)
for info, block in bia.iter_read_block(bchunkshape):
    sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
    bzarr[sl] = block[:]

scheduler = "single-threaded" if NTHREADS == 1 else "threads"

mkl_set_num_threads(1)


@profile
def dask_matmul(azarr, bzarr):
    with dask.config.set(scheduler=scheduler, num_workers=NTHREADS):
        ad = da.from_zarr(azarr)
        bd = da.from_zarr(bzarr)
        cd = da.matmul(ad, bd)
        czarr = zarr.empty(
            (ashape[0], bshape[1]),
            dtype=DTYPE,
            compressor=compressor,
            chunks=(ablock[0], bblock[1]),
        )
        da.to_zarr(cd, czarr)
        return czarr


t0 = time()
czarr = dask_matmul(azarr, bzarr)
t1 = time()
tzarr = t1 - t0
print("Time for computing matmul (via dask): %.3f" % (tzarr))
print(f"a cratio: {azarr.nbytes / azarr.nbytes_stored}")
print(f"b cratio: {bzarr.nbytes / bzarr.nbytes_stored}")
print(f"out cratio: {czarr.nbytes / czarr.nbytes_stored}")

np1 = ia.iarray2numpy(cia)
np2 = np.array(czarr)
np.testing.assert_allclose(np1, np2, rtol=1e-5)

anp = ia.iarray2numpy(aia)
bnp = ia.iarray2numpy(bia)


@profile
def ia_matmul(anp, bnp):
    mkl_set_num_threads(NTHREADS)
    return np.matmul(anp, bnp)


t0 = time()
cia = ia_matmul(anp, bnp)
t1 = time()
tnp = t1 - t0

print("Time for computing matmul (via numpy): %.3f" % tnp)
print(f"Speed-up vs zarr: {tzarr / tia}")
print(f"Speed-up vs numpy (MKL): {tnp / tia}")
