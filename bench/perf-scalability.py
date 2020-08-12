import perfplot
import numpy as np
import iarray as ia


NTHREADS = 4
CHUNKSHAPE = 100 * 1000
BLOCKSHAPE = 16 * 1000


def evaluate(command):
    iax, iay, iaz = (None,) * 3
    shape, chunkshape, blockshape, dtype, cparams = (None,) * 5

    def setup(n):
        # iarray
        global iax, iay, iaz, shape, chunkshape, blockshape, dtype, cparams
        shape = [n]
        chunkshape = [CHUNKSHAPE]
        blockshape = [BLOCKSHAPE]
        dtype = np.float64
        cparams = dict(clib=ia.LZ4, clevel=5, nthreads=NTHREADS)

        iax = ia.linspace(ia.dtshape(shape, dtype), 0, 1,
                          storage=ia.StorageProperties("blosc", chunkshape, blockshape), **cparams)
        iay = iax.copy(storage=ia.StorageProperties("blosc", chunkshape, blockshape), **cparams)
        iaz = iax.copy(storage=ia.StorageProperties("blosc", chunkshape, blockshape), **cparams)

        return command

    def ia_llvm_parallel(command, nthreads):
        global iax, iay, iaz, shape, chunkshape, blockshape, dtype, cparams
        cparams['nthreads'] = nthreads
        eval_flags = ia.EvalFlags(method="iterblosc2", engine="compiler")
        expr = ia.Expr(eval_flags=eval_flags, **cparams)
        expr.bind('x', iax)
        expr.bind('y', iay)
        expr.bind('z', iaz)
        expr.bind_out_properties(ia.dtshape(shape, dtype), ia.StorageProperties("blosc", chunkshape, blockshape))
        expr.compile(command)
        expr.eval()

    perfplot.show(
        setup=setup,
        # n_range=[int(k) for k in range(int(1e8), int(2e8), int(3e7))],
        n_range=[int(k) for k in range(int(1e5), int(1e6), int(1e5))],
        kernels=[lambda x: ia_llvm_parallel(x, 1),
                 lambda x: ia_llvm_parallel(x, 2),
                 lambda x: ia_llvm_parallel(x, 3),
                 lambda x: ia_llvm_parallel(x, 4),
                 ],
        labels=["nthreads=1",
                "nthreads=2",
                "nthreads=3",
                "nthreads=4",
                ],
        logx=False,
        logy=False,
        title="Scalability for iarray with LLVM + iterblosc (3 operands)",
        xlabel='len(x)',
        equality_check=None,
        flops=lambda n: 5 * n,
    )


# evaluate("(x - 1.35) * (x - 4.45) * (x - 8.5)")
evaluate("(x - 1.35) * (y - 4.45) * (z - 8.5)")
