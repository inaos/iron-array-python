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
CODEC = ia.Codecs.LZ4
CLEVEL = 6

FUNCS = ["max", "min", "sum", "prod", "mean"]

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

if os.path.exists("iarray_reduce.iarray"):
    aia = ia.load("iarray_reduce.iarray", load_in_mem=False)
else:
    astorage = ia.Storage(achunkshape, ablockshape, filename="iarray_reduce.iarray")
    dtshape = ia.DTShape(ashape, dtype=DTYPE)
    aia = ia.irandom.normal(dtshape, 0, 1, storage=astorage)

print(f"iarray cratio: {aia.cratio}")


ccompressor = Blosc(
    cname="lz4",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    blocksize=reduce(lambda x, y: x * y, cblockshape) * 8,
)

if not os.path.exists("zarr_reduce.zarr"):
    azarr = zarr.open(
        "zarr_reduce.zarr",
        "w",
        shape=ashape,
        chunks=achunkshape,
        dtype=DTYPE,
        compressor=acompressor,
    )
    for info, block in aia:
        azarr[info.slice] = block[:]
else:
    azarr = zarr.open("zarr_reduce.zarr", "r")


np.testing.assert_equal(ashape, azarr.shape)
print(f"zarr cratio: {azarr.nbytes / azarr.nbytes_stored}")


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
