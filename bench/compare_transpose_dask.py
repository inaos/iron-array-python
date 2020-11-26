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
NTHREADS = 1
CLEVEL = 9
CODEC = ia.Codecs.LZ4

t_iarray = []
t_dask = []
t_ratio = []

ashape = (10000, 10000)
achunkshape = (500, 500)
ablockshape = (128, 128)

cchunkshape = (1000, 1000)
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
nia = ia.random.normal(
    dtshape,
    0,
    0.0000001,
    storage=astorage,
)
aia = (lia + nia).eval(storage=astorage)

cstorage = ia.Storage(cchunkshape, cblockshape)


def ia_transpose(aia):
    return aia.T.copy(cstorage)


mkl_set_num_threads(1)
t0 = time()
cia = ia_transpose(aia)
t1 = time()
tia = t1 - t0
print("Time for computing matmul (via iarray): %.3f" % tia)
print(f"a cratio: {aia.cratio}")
print(f"trans cratio: {cia.cratio}")

azarr = zarr.empty(shape=ashape, chunks=achunkshape, dtype=DTYPE, compressor=compressor)
for info, block in aia.iter_read_block(achunkshape):
    sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
    azarr[sl] = block[:]


scheduler = "single-threaded" if NTHREADS == 1 else "threads"


def dask_transpose(azarr):
    with dask.config.set(scheduler=scheduler, num_workers=NTHREADS):
        ad = da.from_zarr(azarr)
        cd = da.transpose(ad)
        czarr = zarr.empty(
            (ashape[1], ashape[0]),
            dtype=DTYPE,
            compressor=compressor,
            chunks=cchunkshape,
        )
        da.to_zarr(cd, czarr)
        return czarr


mkl_set_num_threads(1)
t0 = time()
czarr = dask_transpose(azarr)
t1 = time()
tzdask = t1 - t0
print("Time for computing matmul (via dask): %.3f" % (tzdask))
print(f"a cratio: {azarr.nbytes / azarr.nbytes_stored}")
print(f"trans cratio: {czarr.nbytes / czarr.nbytes_stored}")

np1 = ia.iarray2numpy(cia)
np2 = np.array(czarr)
np.testing.assert_allclose(np1, np2, rtol=1e-8)

anp = ia.iarray2numpy(aia)


def np_transpose(anp):
    mkl_set_num_threads(NTHREADS)
    return anp.T.copy()


t0 = time()
cia = np_transpose(anp)
t1 = time()
tnp = t1 - t0

print("Time for computing matmul (via numpy): %.3f" % tnp)
print(f"Speed-up vs dask: {tzdask / tia}")
print(f"Speed-up vs numpy (MKL): {tnp / tia}")
