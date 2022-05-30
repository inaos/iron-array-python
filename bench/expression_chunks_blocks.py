# This uses the binary code in LLVM .bc file for evaluating expressions.  Only meant for developers, really.

import iarray as ia
import numpy
import numpy as np
from time import time


def te_expression(method, shape, xcfg, ycfg, zcfg, dtype, expression):
    x = ia.linspace(shape, 0.1, 0.2, dtype=dtype, cfg=xcfg)
    y = ia.linspace(shape, 0, 1, dtype=dtype, cfg=ycfg)
    npx = ia.iarray2numpy(x)
    npy = ia.iarray2numpy(y)

    expr = ia.expr_from_string(expression, {"x": x, "y": y}, cfg=zcfg, eval_method=method)
    t0 = time()
    iout = expr.eval()
    t1 = time()
    print("Eval cfg", iout.cfg)
    print("Evaluation time:", round(t1 - t0, 4))

    npout = ia.iarray2numpy(iout)

    # Evaluate using a different engine (numpy)
    ufunc_repls = {
        "asin": "arcsin",
        "acos": "arccos",
        "atan": "arctan",
        "atan2": "arctan2",
        "pow": "power",
    }
    for ufunc in ufunc_repls.keys():
        if ufunc in expression:
            if ufunc == "pow" and "power" in expression:
                # Don't do a replacement twice
                break
            expression = expression.replace(ufunc, ufunc_repls[ufunc])
    for ufunc in ia.MATH_FUNC_LIST:
        if ufunc in expression:
            idx = expression.find(ufunc)
            # Prevent replacing an ufunc with np.ufunc twice (not terribly solid, but else, test will crash)
            if "np." not in expression[idx - len("np.arc") : idx]:
                expression = expression.replace(ufunc + "(", "np." + ufunc + "(")
    npout2 = eval(expression, {"x": npx, "y": npy, "np": numpy})

    tol = 1e-6 if dtype is np.float32 else 1e-14
    np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)

    ia.remove_urlpath(xcfg.urlpath)
    ia.remove_urlpath(ycfg.urlpath)
    if zcfg is not None:
        ia.remove_urlpath(zcfg.urlpath)


print("Expression with same chunks and blocks for all operands")
# Parameters
method = ia.Eval.ITERCHUNK
shape = [80, 600, 700]
chunks = [40, 400, 400]
blocks = [10, 100, 100]
dtype = np.float32
expression = "(x - cos(y)) * (sin(x) + y) + 2 * x + y"
xcontiguous = False
xurlpath = "test_expression_xsparse.iarr"
ycontiguous = True
yurlpath = None

ia.remove_urlpath(xurlpath)
ia.remove_urlpath(yurlpath)
ia.remove_urlpath("test_expression_zarray.iarr")

xcfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=xcontiguous, urlpath=xurlpath)
ycfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=ycontiguous, urlpath=yurlpath)

te_expression(
    method=method, shape=shape, xcfg=xcfg, ycfg=ycfg, zcfg=None, dtype=dtype, expression=expression
)


print("Expression with different chunks anb blocks for the operands")
# Parameters
method = ia.Eval.ITERCHUNK
dtype = np.float32
expression = "(x - cos(y)) * (sin(x) + y) + 2 * x + y"
xcontiguous = False
xurlpath = "test_expression_xsparse.iarr"
ycontiguous = True
yurlpath = None

ia.remove_urlpath(xurlpath)
ia.remove_urlpath(yurlpath)
ia.remove_urlpath("test_expression_zarray.iarr")

xcfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=xcontiguous, urlpath=xurlpath)
ycfg = ia.Config(chunks=[40, 300, 300], blocks=blocks, contiguous=ycontiguous, urlpath=yurlpath)

print("config defaults ", ia.get_config_defaults())
te_expression(
    method=method, shape=shape, xcfg=xcfg, ycfg=ycfg, zcfg=None, dtype=dtype, expression=expression
)


print("Expression with cfg parameters for the output")
# Parameters
method = ia.Eval.ITERCHUNK
dtype = np.float32
expression = "(x - cos(y)) * (sin(x) + y) + 2 * x + y"
xcontiguous = False
xurlpath = "test_expression_xsparse.iarr"
ycontiguous = True
yurlpath = None

ia.remove_urlpath(xurlpath)
ia.remove_urlpath(yurlpath)
ia.remove_urlpath("test_expression_zarray.iarr")

xcfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=xcontiguous, urlpath=xurlpath)
ycfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=ycontiguous, urlpath=yurlpath)
zcfg = ia.Config(
    chunks=[40, 300, 300],
    blocks=[10, 100, 100],
    contiguous=xcontiguous,
    urlpath="test_expression_zarray.iarr",
)

te_expression(
    method=method, shape=shape, xcfg=xcfg, ycfg=ycfg, zcfg=zcfg, dtype=dtype, expression=expression
)
