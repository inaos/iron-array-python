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


DTYPE = np.float32
NTHREADS = 4
CLEVEL = 9
CODEC = ia.Codecs.LZ4

t_iarray = []
t_dask = []
t_ratio = []

ashape = (2000, 2000)
achunkshape = (250, 250)
ablockshape = (256, 256)


cchunkshape = (250,)
cblockshape = (100,)

axis = 0

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


cstorage = ia.Storage(cchunkshape, cblockshape)


@profile
def ia_reduce(aia):
    return ia.mean(aia, axis=axis, storage=cstorage)


mkl_set_num_threads(1)
t0 = time()
cia = ia_reduce(aia)
t1 = time()
tia = t1 - t0
print("Time for computing reduction (via iarray): %.3f" % tia)
print(f"a cratio: {aia.cratio}")
print(f"out cratio: {cia.cratio}")

azarr = zarr.empty(shape=ashape, chunks=achunkshape, dtype=DTYPE, compressor=compressor)
for info, block in aia.iter_read_block(achunkshape):
    sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
    azarr[sl] = block[:]


scheduler = "single-threaded" if NTHREADS == 1 else "threads"


@profile
def dask_reduce(azarr):
    with dask.config.set(scheduler=scheduler, num_workers=NTHREADS):
        ad = da.from_zarr(azarr)
        cd = da.mean(ad, axis=0)
        czarr = zarr.empty(
            tuple([s for i, s in enumerate(ashape) if i != axis]),
            dtype=DTYPE,
            compressor=compressor,
            chunks=cchunkshape,
        )
        da.to_zarr(cd, czarr)
        return czarr


mkl_set_num_threads(1)
t0 = time()
czarr = dask_reduce(azarr)
t1 = time()
tzdask = t1 - t0
print("Time for computing reduction (via dask): %.3f" % (tzdask))
print(f"a cratio: {azarr.nbytes / azarr.nbytes_stored}")
print(f"out cratio: {czarr.nbytes / czarr.nbytes_stored}")

np1 = ia.iarray2numpy(cia)
np2 = np.array(czarr)
np.testing.assert_allclose(np1, np2, rtol=1e-5)

anp = ia.iarray2numpy(aia)


@profile
def np_reduce(anp):
    mkl_set_num_threads(NTHREADS)
    return np.mean(anp, axis=axis)


t0 = time()
cia = np_reduce(anp)
t1 = time()
tnp = t1 - t0

print("Time for computing reduction (via numpy): %.3f" % tnp)
print(f"Speed-up vs dask: {tzdask / tia}")
print(f"Speed-up vs numpy (MKL): {tnp / tia}")
