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
CLEVEL = 9
CODEC = ia.Codec.LZ4

t_iarray = []
t_dask = []
t_ratio = []

ashape = (10000, 10000)
achunks = (500, 500)
ablocks = (128, 128)

bshape = (10000, 8000)
bchunks = (250, 500)
bblocks = (128, 128)

cchunks = (500, 500)
cblocks = (128, 128)


compressor = Blosc(
    cname="lz4",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    blocksize=reduce(lambda x, y: x * y, ablocks),
)
ia.set_config_defaults(codec=CODEC, clevel=CLEVEL, nthreads=NTHREADS, dtype=DTYPE, btune=False)

acfg = ia.Config(chunks=achunks, blocks=ablocks)
lia = ia.linspace(0, 1, int(np.prod(ashape)), shape=ashape, cfg=acfg)
nia = ia.random.normal(
    ashape,
    0,
    0.0000001,
    cfg=acfg,
)
aia = (lia + nia).eval(cfg=acfg)

bcfg = ia.Config(chunks=bchunks, blocks=bblocks)
lia = ia.linspace(0, 1, int(np.prod(bshape)), shape=bshape, cfg=bcfg)
nia = ia.random.normal(bshape, 0, 0.0000001, cfg=bcfg)
bia = (lia + nia).eval(cfg=bcfg)

ccfg = ia.Config(chunks=cchunks, blocks=cblocks)


@profile
def ia_matmul(aia, bia):
    return ia.matmul(aia, bia, cfg=ccfg)


mkl_set_num_threads(1)
t0 = time()
cia = ia_matmul(aia, bia)
t1 = time()
tia = t1 - t0
print("Time for computing matmul (via iarray): %.3f" % tia)
print(f"a cratio: {aia.cratio}")
print(f"b cratio: {bia.cratio}")
print(f"out cratio: {cia.cratio}")

azarr = zarr.empty(shape=ashape, chunks=achunks, dtype=DTYPE, compressor=compressor)
for info, block in aia.iter_read_block(achunks):
    sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
    azarr[sl] = block[:]


bzarr = zarr.empty(shape=bshape, chunks=bchunks, dtype=DTYPE, compressor=compressor)
for info, block in bia.iter_read_block(bchunks):
    sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
    bzarr[sl] = block[:]

scheduler = "single-threaded" if NTHREADS == 1 else "threads"


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
            chunks=cchunks,
        )
        da.to_zarr(cd, czarr)
        return czarr


mkl_set_num_threads(1)
t0 = time()
czarr = dask_matmul(azarr, bzarr)
t1 = time()
tzdask = t1 - t0
print("Time for computing matmul (via dask): %.3f" % (tzdask))
print(f"a cratio: {azarr.nbytes / azarr.nbytes_stored}")
print(f"b cratio: {bzarr.nbytes / bzarr.nbytes_stored}")
print(f"out cratio: {czarr.nbytes / czarr.nbytes_stored}")

np1 = ia.iarray2numpy(cia)
np2 = np.array(czarr)
np.testing.assert_allclose(np1, np2, rtol=1e-5)

anp = ia.iarray2numpy(aia)
bnp = ia.iarray2numpy(bia)


@profile
def np_matmul(anp, bnp):
    mkl_set_num_threads(NTHREADS)
    return np.matmul(anp, bnp)


t0 = time()
cia = np_matmul(anp, bnp)
t1 = time()
tnp = t1 - t0

print("Time for computing matmul (via numpy): %.3f" % tnp)
print(f"Speed-up vs dask: {tzdask / tia}")
print(f"Speed-up vs numpy (MKL): {tnp / tia}")
