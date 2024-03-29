# Compare reduction performance against dask-zarr

from time import time
from functools import reduce
import dask
import dask.array as da
from numcodecs import Blosc, blosc
import numpy as np
import zarr
import iarray as ia
import gc
import os

DTYPE = np.float64
# Strangely enough, when NTHREADS < 10 there is a significant drop in
# performance (1.1s vs 1.7s) for the OoC situation (at least on a Mac Mini)
NTHREADS = 10
# Using a codec like BLOSCLZ and medium clevel is better here,
# but let's use LZ4 for uniformity
CODEC = ia.Codec.LZ4
CLEVEL = 6

FUNCS = ["max", "min", "sum", "prod", "mean"]

ashape = (27918, 25560)
# These chunks/blocks has been chosen as a balance performance
# between iarray and dask.  In general reducing these values improves
# performance when in memory, but degrades performance when on-disk.
achunks = (2000, 2000)
ablocks = (200, 200)

axis = 0

cshape = tuple([s for i, s in enumerate(ashape) if i != axis])
cchunks = tuple([s for i, s in enumerate(achunks) if i != axis])
cblocks = tuple([s for i, s in enumerate(ablocks) if i != axis])


blosc.use_threads = False
acompressor = Blosc(
    cname="lz4",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    blocksize=reduce(lambda x, y: x * y, ablocks) * 8,
)

ia.set_config_defaults(codec=CODEC, clevel=CLEVEL, nthreads=NTHREADS, btune=False)
print("iA config: ", ia.get_config_defaults())

if os.path.exists("iarray_reduce.iarr"):
    aia = ia.open("iarray_reduce.iarr")
else:
    acfg = ia.Config(chunks=achunks, blocks=ablocks, urlpath="iarray_reduce.iarr")
    aia = ia.random.normal(ashape, 0, 1, cfg=acfg, dtype=DTYPE)

print(f"iarray cratio: {aia.cratio}")


ccompressor = Blosc(
    cname="lz4",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    blocksize=reduce(lambda x, y: x * y, cblocks) * 8,
)

if not os.path.exists("zarr_reduce.zarr"):
    azarr = zarr.open(
        "zarr_reduce.zarr",
        "w",
        shape=ashape,
        chunks=achunks,
        dtype=DTYPE,
        compressor=acompressor,
    )
    for info, block in aia:
        azarr[info.slice] = block[:]
else:
    azarr = zarr.open("zarr_reduce.zarr", "r")

np.testing.assert_equal(ashape, azarr.shape)
print(f"zarr cratio: {azarr.nbytes / azarr.nbytes_stored}")

scheduler = "single-threaded" if NTHREADS == 1 else "threads"

MEMPROF = True
if MEMPROF:
    from memory_profiler import profile
else:

    def profile(f):
        return f


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
            chunks=cchunks,
        )
        da.to_zarr(cd, czarr)
        return czarr


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

    np2 = ia.iarray2numpy(cia)
    np3 = np.asarray(cda)

    np.testing.assert_allclose(np2, np3)
