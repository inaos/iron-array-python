from time import time
import numpy as np

import dask
import dask.array as da
import zarr
from numcodecs import Blosc

import iarray as ia
import matplotlib.pyplot as plt


DTYPE = np.float64
NTHREADS = 8
CLEVEL = 5
CLIB = ia.LZ4

compressor = Blosc(cname='lz4', clevel=CLEVEL, shuffle=Blosc.SHUFFLE)
# shape = (1 * 1000 * 1000,)
shapes = np.logspace(6, 8, 10, dtype=np.int64)
# chunksizes = np.linspace(10 * 1000, 1000 * 1000, 10)
chunkshape = (1000 * 1000,)
blockshape = (16 * 1000,)

sexpr = "(x - 1.35) * (x - 4.45) * (x - 8.5)"

t_iarray = []
t_dask = []
t_ratio = []

for i, shape in enumerate(shapes):
    shape = (shape,)
    print(shape, chunkshape)
    cparams = dict(clib=CLIB, clevel=CLEVEL, nthreads=NTHREADS)

    storage_in = ia.StorageProperties(chunkshape, blockshape, "iarray_infile.iarray")
    ia.arange(ia.dtshape(shape, DTYPE),
              storage=storage_in,
              **cparams)

    t0 = time()
    data = ia.load("iarray_infile.iarray", load_in_mem=False)

    eval_method = ia.EVAL_ITERBLOSC
    expr = ia.Expr(eval_method=eval_method, **cparams)
    expr.bind("x", data)
    storage_out = ia.StorageProperties(chunkshape, blockshape, "iarray_outfile.iarray")
    expr.bind_out_properties(ia.dtshape(shape, DTYPE), storage=storage_out)
    expr.compile(sexpr)
    res1 = expr.eval()
    t1 = time()
    t_iarray.append(t1 - t0)
    print("Time for computing '%s' expression (via ia.Expr()): %.3f" % (sexpr, (t1 - t0)))

    data2 = zarr.open("zarr_infile.zarr", "w", shape=shape, chunks=chunkshape, dtype=DTYPE, compressor=compressor)
    for info, block in data.iter_read_block():
        sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
        data2[sl] = block[:]

    scheduler = "single-threaded" if NTHREADS == 1 else "threads"
    t0 = time()
    with dask.config.set(scheduler=scheduler):
        data2 = zarr.open("zarr_infile.zarr")
        d = da.from_zarr(data2)
        # res = (np.sin(d) - 3.2) * (np.cos(d) + 1.2)
        # res = ((d) - 3.2) * ((d) + 1.2)
        res = (d - 1.35) * (d - 4.45) * (d - 8.5)
        # z2 = zarr.empty(shape, dtype=DTYPE, compressor=compressor, chunks=pshape)
        z2 = zarr.open("zarr_outfile.zarr", "w", shape=shape, chunks=chunkshape,
                       dtype=DTYPE, compressor=compressor)
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
