from time import time
from functools import reduce
import dask
import dask.array as da
from numcodecs import Blosc, blosc
import numpy as np
import zarr
import iarray as ia


DTYPE = np.float64
NTHREADS = 8
CLEVEL = 9
CODEC = ia.Codecs.LZ4

t_iarray = []
t_dask = []
t_ratio = []

ashape = (10000, 10000)
achunkshape = (500, 500)
ablockshape = (100, 100)


cchunkshape = (500,)
cblockshape = (100,)

axis = 0

blosc.use_threads = False
acompressor = Blosc(
    cname="lz4",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    blocksize=reduce(lambda x, y: x * y, ablockshape) * 8,
)


ia.set_config(codec=CODEC, clevel=CLEVEL, nthreads=NTHREADS)

astorage = ia.Storage(achunkshape, ablockshape)
dtshape = ia.DTShape(ashape, dtype=DTYPE)
lia = ia.arange(dtshape, 0, np.prod(ashape), 1, storage=astorage)

nia = ia.random_normal(
    dtshape,
    0,
    0.0000001,
    storage=astorage,
)
aia = (lia + nia).eval(dtshape, storage=astorage)

ccompressor = Blosc(
    cname="lz4",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    blocksize=reduce(lambda x, y: x * y, cblockshape) * 8,
)
cstorage = ia.Storage(cchunkshape, cblockshape)


# @profile
def ia_reduce(aia):
    return ia.sum(aia, axis=axis, storage=cstorage)


t0 = time()
cia = ia_reduce(aia)
t1 = time()
tia = t1 - t0
print("Time for computing reduction (via iarray): %.3f" % tia)
print(f"a cratio: {aia.cratio}")
print(f"out cratio: {cia.cratio}")

azarr = zarr.empty(shape=ashape, chunks=achunkshape, dtype=DTYPE, compressor=acompressor)
for info, block in aia.iter_read_block(achunkshape):
    sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
    azarr[sl] = block[:]


scheduler = "single-threaded" if NTHREADS == 1 else "threads"


# @profile
def dask_reduce(azarr):
    with dask.config.set(scheduler=scheduler, num_workers=NTHREADS):
        ad = da.from_zarr(azarr)
        cd = da.sum(ad, axis=axis)
        czarr = zarr.empty(
            tuple([s for i, s in enumerate(ashape) if i != axis]),
            dtype=DTYPE,
            compressor=ccompressor,
            chunks=cchunkshape,
        )
        da.to_zarr(cd, czarr)
        return czarr


t0 = time()
czarr = dask_reduce(azarr)
t1 = time()
tzdask = t1 - t0
print("Time for computing reduction (via dask): %.3f" % (tzdask))
print(f"a cratio: {azarr.nbytes / azarr.nbytes_stored}")
print(f"out cratio: {czarr.nbytes / czarr.nbytes_stored}")

np1 = ia.iarray2numpy(cia)
np2 = np.array(czarr)

# np.testing.assert_allclose(np1, np2, atol=1e-14, rtol=1e-14)


anp = ia.iarray2numpy(aia)


# @profile
def np_reduce(anp):
    return np.sum(anp, axis=axis)


t0 = time()
cia = np_reduce(anp)
t1 = time()
tnp = t1 - t0

print("Time for computing reduction (via numpy): %.3f" % tnp)
print(f"Speed-up vs dask: {tzdask / tia}")
print(f"Speed-up vs numpy: {tnp / tia}")
