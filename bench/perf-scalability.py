import perfplot
import numpy as np
import iarray as ia


NTHREADS = 20
PSHAPE = 4 * 1000 * 1000


def evaluate(command):
    iax, iay, iaz = (None,) * 3
    shape, pshape, dtype, cparams = (None,) * 4

    def setup(n):
        # iarray
        global iax, iay, iaz, shape, pshape, dtype, cparams
        shape = [n]
        pshape = [PSHAPE]
        dtype = np.float64
        cparams = dict(clib=ia.LZ4, clevel=5, nthreads=NTHREADS)  # , blocksize=1024)
        iax = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 1, **cparams)
        iay = iax.copy(**cparams)
        iaz = iax.copy(**cparams)

        return command

    def ia_llvm_parallel(command, nthreads):
        global iax, iay, iaz, shape, pshape, dtype, cparams
        cparams['nthreads'] = nthreads
        expr = ia.Expr(eval_flags="iterblosc2", **cparams)
        expr.bind('x', iax)
        expr.bind('y', iay)
        expr.bind('z', iaz)
        expr.compile(command)
        expr.eval(shape, pshape, dtype)


    perfplot.show(
        setup=setup,
        # n_range=[int(k) for k in range(int(1e8), int(2e8), int(3e7))],
        n_range=[int(k) for k in range(int(1e7), int(2e8), int(1e7))],
        kernels=[lambda x: ia_llvm_parallel(x, 1),
                 lambda x: ia_llvm_parallel(x, 2),
                 lambda x: ia_llvm_parallel(x, 4),
                 lambda x: ia_llvm_parallel(x, 8),
                 lambda x: ia_llvm_parallel(x, 12),
                 lambda x: ia_llvm_parallel(x, 16),
                 lambda x: ia_llvm_parallel(x, 20),
                 ],
        labels=[command + f" nthreads=1",
                command + f" nthreads=2",
                command + f" nthreads=4",
                command + f" nthreads=8",
                command + f" nthreads=12",
                command + f" nthreads=16",
                command + f" nthreads=20",
                ],
        logx=False,
        logy=False,
        automatic_order=False,
        title="Scalability for iarray with LLVM + iterblosc (3 operands)",
        xlabel='len(x)',
        equality_check=None,
        flops=lambda n: 5 * n,
    )


# evaluate("(x - 1.35) * (x - 4.45) * (x - 8.5)")
evaluate("(x - 1.35) * (y - 4.45) * (z - 8.5)")
