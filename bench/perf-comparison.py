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
CHUNKSHAPE = 100 * 1000
BLOCKSHAPE = 64 * 1000


def evaluate(command):
    x, y, z = (None,) * 3
    iax, iay, iaz = (None,) * 3
    shape, chunkshape, blockshape, dtype, cparams = (None,) * 5
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
        chunkshape = [CHUNKSHAPE]
        blockshape = [BLOCKSHAPE]
        dtype = np.float64
        cparams = dict(clib=ia.LZ4, clevel=5, nthreads=NTHREADS)
        storage = ia.StorageProperties("blosc", chunkshape, blockshape)
        iax = ia.linspace(ia.dtshape(shape, dtype), 0, 1, storage=storage, **cparams)
        iay = ia.linspace(ia.dtshape(shape, dtype), 0, 1, storage=storage, **cparams)
        iaz = ia.linspace(ia.dtshape(shape, dtype), 0, 1, storage=storage, **cparams)
        # dask/zarr
        global zx, zy, zz, zcompr
        zcompr = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE,
                       blocksize=int(np.prod(blockshape) * np.dtype(dtype).itemsize))

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
        cparams['nthreads'] = NTHREADS
        eval_method = ia.EVAL_ITERBLOSC
        expr = ia.Expr(eval_method=eval_method, **cparams)
        expr.bind('x', iax)
        expr.bind('y', iay)
        expr.bind('z', iaz)
        expr.bind_out_properties(ia.dtshape(shape, dtype), ia.StorageProperties("blosc", chunkshape, blockshape))
        expr.compile(command)
        expr.eval()

    def ia_compiler_serial(command):
        global iax, iay, iaz, shape, chunkshape, blockshape, dtype, cparams
        cparams['nthreads'] = 1
        eval_method = ia.EVAL_ITERBLOSC
        expr = ia.Expr(eval_method=eval_method, **cparams)
        expr.bind('x', iax)
        expr.bind('y', iay)
        expr.bind('z', iaz)
        expr.bind_out_properties(ia.dtshape(shape, dtype), ia.StorageProperties("blosc", chunkshape, blockshape))
        expr.compile(command)
        expr.eval()

    # def dask_parallel(command):
    #     global zx, zy, zz, shape, chunkshape, dtype, zcompr
    #     with dask.config.set({"scheduler": "threads", "pool": ThreadPool(NTHREADS)}):
    #         da.from_zarr(zx)
    #         da.from_zarr(zy)
    #         da.from_zarr(zz)
    #         res = eval(command)
    #         zout = zarr.empty(shape, dtype=dtype, compressor=zcompr, chunks=chunkshape)
    #         da.to_zarr(res, zout)
    #
    # def dask_serial(command):
    #     global zx, zy, zz, shape, chunkshape, dtype, zcompr
    #     with dask.config.set(scheduler="single-threaded"):
    #         da.from_zarr(zx)
    #         da.from_zarr(zy)
    #         da.from_zarr(zz)
    #         res = eval(command)
    #         zout = zarr.empty(shape, dtype=dtype, compressor=zcompr, chunks=chunkshape)
    #         da.to_zarr(res, zout)

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
            # dask_parallel,
            # dask_serial,
        ],
        labels=[command + " numpy",
                command + " numexpr parallel",
                command + " numexpr serial",
                command + " iarray parallel (compiler)",
                command + " iarray serial (compiler)",
                # command + " dask parallel (threads)",
                # command + " dask serial",
                ],
        logx=False,
        logy=False,
        title="Comparison with other engines (3 operands, nthreads=%s)" % NTHREADS,
        xlabel='len(x)',
        equality_check=None,
        flops=lambda n: 5 * n,
    )


evaluate("(x - 1.35) * (y - 4.45) * (z - 8.5)")
