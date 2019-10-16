from time import time

import dask
import dask.array as da
import zarr
import numpy as np
from numcodecs import Blosc

import iarray as ia
import matplotlib.pyplot as plt


DTYPE = np.float64
NTHREADS = 4
CLEVEL = 5
CLIB = ia.LZ4

compressor = Blosc(cname='lz4', clevel=CLEVEL, shuffle=Blosc.SHUFFLE)
shape = (50 * 1000 * 1000,)
chunksizes = np.linspace(10 * 1000, 1000 * 1000, 10)

t_iarray = []
t_dask = []

for i, chunksize in enumerate(chunksizes):
    print(chunksize)

    pshape = tuple([chunksize for s in shape])

    ia.arange(ia.dtshape(shape, pshape=pshape, dtype=DTYPE), filename="iarray_file.iarray", clib=CLIB, clevel=CLEVEL)

    sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"

    t0 = time()
    data = ia.from_file("iarray_file.iarray", load_in_mem=False)
    expr = ia.Expr(eval_flags="iterblock", blocksize=0, nthreads=NTHREADS, clevel=CLEVEL)
    expr.bind("x", data)
    expr.compile(sexpr)
    res1 = expr.eval(shape)
    t1 = time()
    t_iarray.append(t1 - t0)
    print("Time for computing '%s' expression (via ia.Expr()): %.3f" % (sexpr, (t1 - t0)))

    data2 = zarr.open("zarr_file.zarr", "w", shape=shape, chunks=pshape, dtype=DTYPE, compressor=compressor)
    for info, block in data.iter_read_block():
        sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
        data2[sl] = block[:]

    scheduler = "single-threaded" if NTHREADS == 1 else "threads"
    t0 = time()
    with dask.config.set(scheduler=scheduler):
        data2 = zarr.open("zarr_file.zarr")
        d = da.from_zarr(data2)
        res = (np.sin(d) - 3.2) * (np.cos(d) + 1.2)
        z2 = zarr.empty(shape, dtype=DTYPE, compressor=compressor)
        da.to_zarr(res, z2)
    t1 = time()
    t_dask.append(t1 - t0)
    print("Time for computing '%s' expression (via dask): %.3f" % (sexpr, (t1 - t0)))

    #np1 = ia.iarray2numpy(res1)
    #np2 = np.array(z2)
    #np.testing.assert_allclose(np1, np2)

    print(f"Speed up: {t_dask[i] / t_iarray[i]:.4f} x")

plt.plot(chunksizes, t_iarray, label='iarray')
plt.plot(chunksizes, t_dask, label="zarr + dask")
plt.legend()
plt.title("On disk")
plt.ylabel("Time (s)")
plt.xlabel("Num. elements")
plt.show()
