import threading
from multiprocessing.pool import ThreadPool
import perfplot
import numpy as np
import iarray as ia
import numexpr as ne
import dask
import dask.array as da
import zarr
from numcodecs import Blosc


NTHREADS = 20
PSHAPE = 4 * 1000 * 1000


def evaluate(command):
    x, y, z = (None,) * 3
    iax, iay, iaz = (None,) * 3
    shape, pshape, dtype, cparams = (None,) * 4
    zx, zy, zz = (None,) * 3
    zcompr = None

    def setup(n):
        # numpy/numexpr
        global x, y, z
        x = np.linspace(0, 1, n)
        y = x.copy()
        z = y.copy()
        # iarray
        global iax, iay, iaz, shape, pshape, dtype, cparams
        shape = [n]
        pshape = [PSHAPE]
        dtype = np.float64
        cparams = dict(clib=ia.LZ4, clevel=5, nthreads=NTHREADS)  # , blocksize=1024)
        iax = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 1, **cparams)
        iay = iax.copy(**cparams)
        iaz = iax.copy(**cparams)
        # dask/zarr
        global zx, zy, zz, zcompr
        zcompr = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
        zx = zarr.empty(shape=shape, chunks=pshape, dtype=dtype, compressor=zcompr)
        zy = zarr.empty(shape=shape, chunks=pshape, dtype=dtype, compressor=zcompr)
        zz = zarr.empty(shape=shape, chunks=pshape, dtype=dtype, compressor=zcompr)
        for info, block in iax.iter_read_block():
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
        global iax, iay, iaz, shape, pshape, dtype, cparams
        cparams['nthreads'] = NTHREADS
        eval_flags = ia.EvalFlags(method="iterblosc2", engine="compiler")
        expr = ia.Expr(eval_flags=eval_flags, **cparams)
        expr.bind('x', iax)
        expr.bind('y', iay)
        expr.bind('z', iaz)
        expr.compile(command)
        expr.eval(shape, pshape, dtype)

    def ia_compiler_serial(command):
        global iax, iay, iaz, shape, pshape, dtype, cparams
        cparams['nthreads'] = 1
        eval_flags = ia.EvalFlags(method="iterblosc2", engine="compiler")
        expr = ia.Expr(eval_flags=eval_flags, **cparams)
        expr.bind('x', iax)
        expr.bind('y', iay)
        expr.bind('z', iaz)
        expr.compile(command)
        expr.eval(shape, pshape, dtype)

    def ia_interpreter_parallel(command):
        global iax, iay, iaz, shape, pshape, dtype, cparams
        cparams['nthreads'] = NTHREADS
        eval_flags = ia.EvalFlags(method="iterblosc2", engine="interpreter")
        expr = ia.Expr(eval_flags=eval_flags, **cparams)
        expr.bind('x', iax)
        expr.bind('y', iay)
        expr.bind('z', iaz)
        expr.compile(command)
        expr.eval(shape, pshape, dtype)

    def ia_interpreter_serial(command):
        global iax, iay, iaz, shape, pshape, dtype, cparams
        cparams['nthreads'] = 1
        eval_flags = ia.EvalFlags(method="iterblosc2", engine="interpreter")
        expr = ia.Expr(eval_flags=eval_flags, **cparams)
        expr.bind('x', iax)
        expr.bind('y', iay)
        expr.bind('z', iaz)
        expr.compile(command)
        expr.eval(shape, pshape, dtype)

    def dask_parallel(command):
        global zx, zy, zz, shape, pshape, dtype, zcompr
        with dask.config.set({"scheduler": "threads", "pool": ThreadPool(NTHREADS)}):
            x = da.from_zarr(zx)
            y = da.from_zarr(zy)
            z = da.from_zarr(zz)
            res = eval(command)
            zout = zarr.empty(shape, dtype=dtype, compressor=zcompr, chunks=pshape)
            da.to_zarr(res, zout)

    def dask_serial(command):
        global zx, zy, zz, shape, pshape, dtype, zcompr
        with dask.config.set(scheduler="single-threaded"):
            x = da.from_zarr(zx)
            y = da.from_zarr(zy)
            z = da.from_zarr(zz)
            res = eval(command)
            zout = zarr.empty(shape, dtype=dtype, compressor=zcompr, chunks=pshape)
            da.to_zarr(res, zout)

    perfplot.show(
        setup=setup,
        n_range=[int(k) for k in range(int(1e7), int(2e8), int(3e7))],
        # n_range=[int(k) for k in range(int(1e7), int(2e8), int(1e7))],
        kernels=[
            np_serial,
            ne_parallel,
            ne_serial,
            ia_compiler_parallel,
            ia_compiler_serial,
            ia_interpreter_parallel,
            ia_interpreter_serial,
            dask_parallel,
            dask_serial,
        ],
        labels=[command + " numpy",
                command + " numexpr parallel",
                command + " numexpr serial",
                command + " iarray parallel (compiler)",
                command + " iarray serial (compiler)",
                command + " iarray parallel (interpreter)",
                command + " iarray serial (interpreter)",
                command + " dask parallel (threads)",
                command + " dask serial",
                ],
        logx=False,
        logy=False,
        title="Comparison with other engines (3 operands, nthreads=%s)" % NTHREADS,
        xlabel='len(x)',
        equality_check=None,
        flops=lambda n: 5 * n,
    )


#evaluate("(x - 1.35) * (x - 4.45) * (x - 8.5)")
evaluate("(x - 1.35) * (y - 4.45) * (z - 8.5)")
