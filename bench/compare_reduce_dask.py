from time import time
from functools import reduce
import dask
import dask.array as da
from numcodecs import Blosc, blosc
import numpy as np
import zarr
import iarray as ia
import gc

DTYPE = np.float64
FUNCS = ["max", "min", "sum", "prod", "mean"]
NTHREADS = 10
# Using a codec like BLOSCLZ and medium clevel is better here,
# but let's use LZ4 for uniformity
CODEC = ia.Codecs.LZ4
CLEVEL = 6

ashape = (27918, 25560)
# These chunkshape/blockshape has been chosen as a balance performance
# between iarray and dask.  In general reducing these values improves
# performance when in memory, but degrades performance when on-disk.
achunkshape = (2000, 2000)
ablockshape = (200, 200)

axis = 0

cshape = tuple([s for i, s in enumerate(ashape) if i != axis])
cchunkshape = tuple([s for i, s in enumerate(achunkshape) if i != axis])
cblockshape = tuple([s for i, s in enumerate(ablockshape) if i != axis])


blosc.use_threads = False
acompressor = Blosc(
    cname="lz4",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    blocksize=reduce(lambda x, y: x * y, ablockshape) * 8,
)

ia.set_config(codec=CODEC, clevel=CLEVEL, nthreads=NTHREADS, fp_mantissa_bits=20)

astorage = ia.Storage(achunkshape, ablockshape)
dtshape = ia.DTShape(ashape, dtype=DTYPE)
aia = ia.random_normal(dtshape, 0, 1, storage=astorage)
print(f"iarray cratio: {aia.cratio}")

ccompressor = Blosc(
    cname="lz4",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    blocksize=reduce(lambda x, y: x * y, cblockshape) * 8,
)

azarr = zarr.empty(shape=ashape, chunks=achunkshape, dtype=DTYPE, compressor=acompressor)
for info, block in aia.iter_read_block(achunkshape):
    sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
    azarr[sl] = block[:]

print(f"zarr cratio: {azarr.nbytes / azarr.nbytes_stored}")

anp = ia.iarray2numpy(aia)
print(f"numpy cratio: 1")

ia.set_config(fp_mantissa_bits=0)
scheduler = "single-threaded" if NTHREADS == 1 else "threads"


@profile
def ia_reduce(aia, func):
    return func(aia, axis=axis)


@profile
def dask_reduce(azarr, func):
    with dask.config.set(scheduler=scheduler, num_workers=NTHREADS):
        ad = da.from_zarr(azarr)
        cd = func(ad, axis=axis)
        czarr = zarr.empty(
            cshape,
            dtype=DTYPE,
            compressor=ccompressor,
            chunks=cchunkshape,
        )
        da.to_zarr(cd, czarr)
        return czarr


@profile
def np_reduce(anp, func):
    return func(anp, axis=axis)


for func in FUNCS:

    print(f"{func}")

    gc.collect()
    t0 = time()
    cia = ia_reduce(aia, getattr(ia, func))
    t1 = time()
    tia = t1 - t0
    print(f"- Time for computing iarray {func}: {tia:.3f} s")
    print(f"  iarray cratio {cia.cratio:.3f}")

    gc.collect()
    t0 = time()
    cda = dask_reduce(azarr, getattr(da, func))
    t1 = time()
    tda = t1 - t0
    print(f"- Time for computing dask {func}: {tda:.3f} s")
    cratio = cda.nbytes / cda.nbytes_stored
    print(f"  zarr cratio {cratio:.3f}")

    gc.collect()
    t0 = time()
    cnp = np_reduce(anp, getattr(np, func))
    t1 = time()
    tnp = t1 - t0
    print(f"- Time for computing numpy {func}: {tnp:.3f} s")

    np1 = ia.iarray2numpy(cia)
    np2 = np.asarray(cda)

    np.testing.assert_allclose(np1, np2)
    np.testing.assert_allclose(np2, cnp)
