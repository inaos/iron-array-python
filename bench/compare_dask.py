from time import time
import numpy as np

import dask
import dask.array as da
import zarr
from numcodecs import Blosc

import iarray as ia
import matplotlib.pyplot as plt


DTYPE = np.float32
NTHREADS = 4
CLEVEL = 5
CLIB = ia.LZ4

compressor = Blosc(cname='lz4', clevel=CLEVEL, shuffle=Blosc.SHUFFLE)
# shape = (1 * 1000 * 1000,)
shapes = np.logspace(5, 8, 10, dtype=np.int64)
# chunksizes = np.linspace(10 * 1000, 1000 * 1000, 10)
pshape = (100 * 1000,)
# pshape = None

sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"

t_iarray = []
t_dask = []
t_ratio = []

for i, shape in enumerate(shapes):
    shape = (shape,)
    print(shape)
    ia.arange(ia.dtshape(shape, pshape=pshape, dtype=DTYPE), filename="iarray_infile.iarray", clib=CLIB, clevel=CLEVEL)

    t0 = time()
    data = ia.from_file("iarray_infile.iarray", load_in_mem=False)
    expr = ia.Expr(eval_flags="iterblock", blocksize=0, nthreads=NTHREADS, clevel=CLEVEL)
    expr.bind("x", data)
    expr.compile(sexpr)
    res1 = expr.eval(shape, pshape=pshape, dtype=DTYPE, filename="iarray_outfile.iarray")
    t1 = time()
    t_iarray.append(t1 - t0)
    print("Time for computing '%s' expression (via ia.Expr()): %.3f" % (sexpr, (t1 - t0)))

    data2 = zarr.open("zarr_infile.zarr", "w", shape=shape, chunks=pshape, dtype=DTYPE, compressor=compressor)
    for info, block in data.iter_read_block():
        sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
        data2[sl] = block[:]

    scheduler = "single-threaded" if NTHREADS == 1 else "threads"
    t0 = time()
    with dask.config.set(scheduler=scheduler):
        data2 = zarr.open("zarr_infile.zarr")
        d = da.from_zarr(data2)
        res = (np.sin(d) - 3.2) * (np.cos(d) + 1.2)
        # z2 = zarr.empty(shape, dtype=DTYPE, compressor=compressor, chunks=pshape)
        z2 = zarr.open("zarr_outfile.zarr", "w", shape=shape, chunks=pshape, dtype=DTYPE, compressor=compressor)
        da.to_zarr(res, z2)
    t1 = time()
    t_dask.append(t1 - t0)
    print("Time for computing '%s' expression (via dask): %.3f" % (sexpr, (t1 - t0)))

    # np1 = ia.iarray2numpy(res1)
    # np2 = np.array(z2)
    # np.testing.assert_allclose(np1, np2)

    t_ratio.append(t_dask[i] / t_iarray[i])
    print(f"Speed up: {t_ratio[i]:.2f}x")

# plt.loglog(shapes, t_iarray, label='iarray')
# plt.loglog(shapes, t_dask, label="zarr + dask")
# plt.semilogx(shapes, t_ratio, label="(zarr + dask) / iarray")
plt.bar(np.log10(shapes), t_ratio, label="t(zarr + dask) / t(iarray)", width=0.07)
plt.legend()
plt.title(f"Times for computing '{sexpr}' (on disk)")
plt.ylabel("Time ratio")
plt.xlabel("log10(elements in array)")
# plt.ylim(bottom=0)
plt.show()