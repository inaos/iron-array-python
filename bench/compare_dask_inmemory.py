from time import time
import numpy as np

import dask
import dask.array as da
from multiprocessing.pool import ThreadPool
import zarr
from numcodecs import Blosc

import iarray as ia
import matplotlib.pyplot as plt


NTHREADS = 8
CLEVEL = 5
CODEC = ia.Codecs.LZ4

shapes = np.logspace(6, 8, 10, dtype=np.int64)
# chunkshape, blockshape = (100_000,), (8_000,)
chunkshape, blockshape = None, None

dtype = np.float64
compressor = Blosc(cname="lz4", clevel=CLEVEL, shuffle=Blosc.SHUFFLE)
cparams = dict(codec=CODEC, clevel=CLEVEL, nthreads=NTHREADS)

sexpr = "(x - 1.35) * (x - 4.45) * (x - 8.5)"

t_iarray = []
t_dask = []
t_ratio = []

for i, shape in enumerate(shapes):
    shape = (shape,)
    print(shape)
    dtshape = ia.DTShape(shape, dtype)
    storage = ia.Storage(chunkshape, blockshape)
    data = ia.linspace(dtshape, 0, 1, storage=storage, **cparams)

    t0 = time()
    expr = ia.expr_from_string(sexpr, {"x": data}, dtshape, **cparams)
    res1 = expr.eval()
    t1 = time()
    t_iarray.append(t1 - t0)

    print("Time for computing '%s' expression (via ia.Expr()): %.3f" % (sexpr, (t1 - t0)))

    data2 = zarr.empty(shape=shape, chunks=chunkshape, dtype=dtype, compressor=compressor)
    for info, block in data.iter_read_block(chunkshape):
        sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
        data2[sl] = block[:]

    scheduler = "single-threaded" if NTHREADS == 1 else "threads"
    t0 = time()
    with dask.config.set(scheduler=scheduler, pool=ThreadPool(NTHREADS)):
        d = da.from_zarr(data2)
        res = (d - 1.35) * (d - 4.45) * (d - 8.5)
        z2 = zarr.empty(shape, dtype=dtype, compressor=compressor, chunks=chunkshape)
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
plt.title(f"Times for computing '{sexpr}' (in memory)")
plt.ylabel("Time ratio")
plt.xlabel("log10(elements in array)")
# plt.ylim(bottom=0)
plt.show()
