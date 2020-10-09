# Scalability plot comparing iarray vs dask for evaluating expressions

import perfplot
import numpy as np
import iarray as ia
import numexpr as ne
import dask
import dask.array as da
from multiprocessing.pool import ThreadPool
import zarr
from numcodecs import Blosc


NTHREADS = 8
# CHUNKSHAPE, BLOCKSHAPE = [1000_000], [16_000]
CHUNKSHAPE, BLOCKSHAPE = None, None  # automatic partitioning
CLEVEL = 5


def evaluate(command):
    x, y, z = (None,) * 3
    iax, iay, iaz = (None,) * 3
    shape, chunkshape, blockshape, dtype = (None,) * 4
    cparams = {}
    zx, zy, zz = (None,) * 3
    zcompr = None

    def setup(n):
        # numpy/numexpr
        global x, y, z
        x = np.linspace(0, 1, n)
        y = x.copy()
        z = y.copy()
        # iarray
        global iax, iay, iaz, shape, chunkshape, blockshape, dtype, cparams
        shape = [n]
        chunkshape = CHUNKSHAPE
        blockshape = BLOCKSHAPE
        dtype = np.float64
        cparams = dict(clevel=CLEVEL, nthreads=NTHREADS)
        storage = ia.Storage(chunkshape, blockshape)
        dtshape = ia.DTShape(shape, dtype)
        iax = ia.linspace(dtshape, 0, 1, storage=storage, **cparams)
        iay = ia.linspace(dtshape, 0, 1, storage=storage, **cparams)
        iaz = ia.linspace(dtshape, 0, 1, storage=storage, **cparams)

        # dask/zarr
        global zx, zy, zz, zcompr
        zcompr = Blosc(clevel=CLEVEL, shuffle=Blosc.SHUFFLE)
        zx = zarr.empty(shape=shape, chunks=chunkshape, dtype=dtype, compressor=zcompr)
        zy = zarr.empty(shape=shape, chunks=chunkshape, dtype=dtype, compressor=zcompr)
        zz = zarr.empty(shape=shape, chunks=chunkshape, dtype=dtype, compressor=zcompr)
        for info, block in iax.iter_read_block(chunkshape):
            sl = tuple([slice(i, i + s) for i, s in zip(info.elemindex, info.shape)])
            zx[sl] = block[:]
            zy[sl] = block[:]
            zz[sl] = block[:]

        return command

    def np_serial(command):
        eval(command)

    def ne_parallel(command):
        ne.set_num_threads(NTHREADS)
        ne.evaluate(command)

    def ne_serial(command):
        ne.set_num_threads(1)
        ne.evaluate(command)

    def ia_compiler_parallel(command):
        global iax, iay, iaz, shape, chunkshape, blockshape, dtype, cparams
        cparams["nthreads"] = NTHREADS
        cparams["storage"] = ia.Storage(chunkshape, blockshape)
        dtshape = ia.DTShape(shape, dtype)
        expr = ia.create_expr(command, {"x": iax, "y": iay, "z": iaz}, dtshape, **cparams)
        expr.eval()

    def ia_compiler_serial(command):
        global iax, iay, iaz, shape, chunkshape, blockshape, dtype, cparams
        cparams["nthreads"] = 1
        cparams["storage"] = ia.Storage(chunkshape, blockshape)
        dtshape = ia.DTShape(shape, dtype)
        expr = ia.create_expr(command, {"x": iax, "y": iay, "z": iaz}, dtshape, **cparams)
        expr.eval()

    def dask_parallel(command):
        global zx, zy, zz, shape, chunkshape, dtype, zcompr
        with dask.config.set(scheduler="threads", pool=ThreadPool(NTHREADS)):
            x = da.from_zarr(zx)
            y = da.from_zarr(zy)
            z = da.from_zarr(zz)
            res = eval(command)
            zout = zarr.empty(shape, dtype=dtype, compressor=zcompr, chunks=chunkshape)
            da.to_zarr(res, zout)

    def dask_serial(command):
        global zx, zy, zz, shape, chunkshape, dtype, zcompr
        with dask.config.set(scheduler="single-threaded"):
            x = da.from_zarr(zx)
            y = da.from_zarr(zy)
            z = da.from_zarr(zz)
            res = eval(command)
            zout = zarr.empty(shape, dtype=dtype, compressor=zcompr, chunks=chunkshape)
            da.to_zarr(res, zout)

    perfplot.show(
        setup=setup,
        # n_range=[int(k) for k in range(int(1e7), int(2e8), int(3e7))],
        # n_range=[int(k) for k in range(int(1e7), int(2e8), int(5e7))],
        n_range=[int(k) for k in range(int(1e7), int(2e8), int(1e7))],
        kernels=[
            np_serial,
            # ne_parallel,
            # ne_serial,
            ia_compiler_parallel,
            ia_compiler_serial,
            dask_parallel,
            dask_serial,
        ],
        labels=[
            "numpy",
            # "numexpr parallel",
            # "numexpr serial",
            "iarray parallel",
            "iarray serial",
            "dask parallel",
            "dask serial",
        ],
        logx=False,
        logy=False,
        title=f"Eval {command} (nthreads={NTHREADS})",
        xlabel="len(x)",
        equality_check=None,
        flops=lambda n: 5 * n,
    )


evaluate("(x - 1.35) * (y - 4.45) * (z - 8.5)")
