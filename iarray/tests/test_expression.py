import pytest
import iarray as ia
import numpy
import numpy as np


# Expression
@pytest.mark.parametrize("method, shape, chunkshape, blockshape, dtype, expression", [
    (ia.Eval.ITERBLOSC, [100, 100], [23, 32], [10, 10], np.float64, "cos(x)"),  # TODO: fix this
    (ia.Eval.ITERBLOSC, [100, 100], [10, 99], [4, 12], np.float64, "x"),
    (ia.Eval.ITERBLOSC, [1000], [110], [55], np.float32, "x"),
    (ia.Eval.ITERBLOSC, [1000], [100], [30], np.float64, "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)"),
    (ia.Eval.AUTO, [1000], [100], [25], np.float64, "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)"),
    (ia.Eval.ITERCHUNK, [1000], [367], [77], np.float32, "(abs(-x) - 1.35) * ceil(x) * floor(x - 8.5)"),
    (ia.Eval.ITERBLOSC, [100, 100, 100], [25, 25, 33], [12, 16, 8], np.float64,
     "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)"),
    (ia.Eval.ITERBLOSC, [223], [100], [30], np.float64, "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)"),
    (ia.Eval.ITERCHUNK, [100, 100, 55], [10, 5, 10], [3, 4, 3], np.float64,
     "asin(x) + (acos(x) - 1.35) - atan(x + .2)"),
    (ia.Eval.ITERCHUNK, [100, 100, 55], [10, 5, 10], [3, 4, 3], np.float64,
     "arcsin(x) + (arccos(x) - 1.35) - arctan(x + .2)"),  # check NumPy naming convention for ufuncs
    (ia.Eval.AUTO, [1000], None, None, np.float64, "exp(x) + (log(x) - 1.35) - log10(x + .2)"),
    (ia.Eval.ITERCHUNK, [1000], None, None, np.float32, "sqrt(x) + atan2(x, x) + pow(x, x)"),
    (ia.Eval.AUTO, [1000], None, None, np.float32, "sqrt(x) + arctan2(x, x) + power(x, x)"),  # NumPy conventions
    (ia.Eval.AUTO, [100, 100], None, None, np.float64, "(x - cos(1)) * 2"),
    (ia.Eval.ITERCHUNK, [8, 6, 7, 4, 5], None, None, np.float32,
     "(x - cos(y)) * (sin(x) + y) + 2 * x + y"),
    (ia.Eval.ITERBLOSC,  [17, 12, 15, 15, 8], [8, 6, 7, 4, 5], [4, 3, 3, 4, 5], np.float64,
     "(x - cos(y)) * (sin(x) + y) + 2 * x + y"),
])
def test_expression(method, shape, chunkshape, blockshape, dtype, expression):
    # The ranges below are important for not overflowing operations
    if chunkshape is None:
        storage = ia.StorageProperties(backend=ia.BACKEND_PLAINBUFFER)
    else:
        storage = ia.StorageProperties(chunkshape=chunkshape, blockshape=blockshape, filename=None, enforce_frame=False,
                                       backend=ia.BACKEND_BLOSC)

    x = ia.linspace(ia.dtshape(shape, dtype), 2.1, .2, storage=storage)
    y = ia.linspace(ia.dtshape(shape, dtype), 0, 1, storage=storage)
    npx = ia.iarray2numpy(x)
    npy = ia.iarray2numpy(y)

    expr = ia.Expr(eval_method=method)
    expr.bind("x", x)
    expr.bind("y", y)
    expr.bind_out_properties(ia.dtshape(shape, dtype), storage=storage)

    expr.compile(expression)

    iout = expr.eval()
    npout = ia.iarray2numpy(iout)

    # Evaluate using a different engine
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
    for ufunc in ia.UFUNC_LIST:
        if ufunc in expression:
            idx = expression.find(ufunc)
            # Prevent replacing an ufunc with np.ufunc twice (not terribly solid, but else, test will crash)
            if "np." not in expression[idx - len("np.arc"):idx]:
                expression = expression.replace(ufunc + "(", "np." + ufunc + "(")
    npout2 = eval(expression, {"x": npx, "y": npy, "np": numpy})

    decimal = 5 if dtype is np.float32 else 10

    np.testing.assert_almost_equal(npout, npout2, decimal=decimal)


# ufuncs
@pytest.mark.parametrize("ufunc, ia_expr", [
    ("abs(x)", "abs(x)"),
    ("arccos(x)", "acos(x)"),
    ("arcsin(x)", "asin(x)"),
    ("arctan(x)", "atan(x)"),
    ("arctan2(x, y)", "atan2(x, y)"),
    ("ceil(x)", "ceil(x)"),
    ("cos(x)", "cos(x)"),
    ("cosh(x)", "cosh(x)"),
    ("exp(x)", "exp(x)"),
    ("floor(x)", "floor(x)"),
    ("log(x)", "log(x)"),
    ("log10(x)", "log10(x)"),
    # ("negative(x)", "negate(x)"),
    ("power(x, y)", "pow(x, y)"),
    ("sin(x)", "sin(x)"),
    ("sinh(x)", "sinh(x)"),
    ("sqrt(x)", "sqrt(x)"),
    ("tan(x)", "tan(x)"),
    ("tanh(x)", "tanh(x)"),
])
def test_ufuncs(ufunc, ia_expr):
    shape = [200, 300]
    chunkshape = [40, 40]
    bshape = [10, 17]

    storage = ia.StorageProperties(chunkshape=chunkshape, blockshape=bshape)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(ia.dtshape(shape, dtype), .1, .9, storage=storage)
        y = ia.linspace(ia.dtshape(shape, dtype), 0, 1, storage=storage)
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)

        # Low-level ironarray eval
        expr = ia.Expr()
        expr.bind("x", x)
        expr.bind("y", y)
        expr.bind_out_properties(ia.dtshape(shape, dtype), storage=storage)
        expr.compile(ia_expr)
        iout = expr.eval()
        npout = ia.iarray2numpy(iout)

        decimal = 6 if dtype is np.float32 else 7

        # High-level ironarray eval
        lazy_expr = eval("ia." + ufunc, {"ia": ia, "x": x, "y": y})
        iout2 = lazy_expr.eval(dtype=dtype)
        npout2 = ia.iarray2numpy(iout2)
        np.testing.assert_almost_equal(npout, npout2, decimal=decimal)

        # High-level ironarray eval, but via numpy ufunc machinery
        # TODO: the next ufuncs still have some problems with the numpy machinery (bug?)
        # abs(x) : TypeError: bad operand type for abs(): 'IArray'
        # ceil(x) : TypeError: must be real number, not IArray
        # floor(x): TypeError: must be real number, not IArray
        # negative(x) : TypeError: bad operand type for unary -: 'IArray'
        # power(x,y) : TypeError: unsupported operand type(s) for ** or pow(): 'IArray' and 'IArray'
        if ufunc not in ("abs(x)", "ceil(x)", "floor(x)", "negative(x)", "power(x, y)"):
            lazy_expr = eval("np." + ufunc, {"np": np, "x": x, "y": y})
            iout2 = lazy_expr.eval(dtype=dtype)
            npout2 = ia.iarray2numpy(iout2)
            np.testing.assert_almost_equal(npout, npout2, decimal=decimal)

        npout2 = eval("np." + ufunc, {"np": np, "x": npx, "y": npy})   # pure numpy
        np.testing.assert_almost_equal(npout, npout2, decimal=decimal)


# ufuncs inside of expressions
@pytest.mark.parametrize("ufunc", [
    "abs",
    "arccos",
    "arcsin",
    "arctan",
    "arctan2",
    "ceil",
    "cos",
    "cosh",
    "exp",
    "floor",
    "log",
    "log10",
    # "negative",
    "power",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
])
def test_expr_ufuncs(ufunc):
    shape = [200, 300]
    chunkshape = [40, 50]
    bshape = [20, 20]
    storage = ia.StorageProperties(chunkshape=chunkshape, blockshape=bshape)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(ia.dtshape(shape, dtype), .1, .9, storage=storage)
        y = ia.linspace(ia.dtshape(shape, dtype), 0, 1, storage=storage)

        # NumPy computation
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)
        if ufunc in ("arctan2", "power"):
            npout = eval("1 + 2 * np.%s(x, y)" % ufunc, {"np": np, "x": npx, "y": npy})
        else:
            npout = eval("1 + 2 * np.%s(x)" % ufunc, {"np": np, "x": npx})

        # High-level ironarray eval
        if ufunc in ("arctan2", "power"):
            lazy_expr = eval("1 + 2* x.%s(y)" % ufunc, {"x": x, "y": y})
        else:
            lazy_expr = eval("1 + 2 * x.%s()" % ufunc, {"x": x})
        iout2 = lazy_expr.eval(dtype=dtype)
        npout2 = ia.iarray2numpy(iout2)

        decimal = 6 if dtype is np.float32 else 7
        np.testing.assert_almost_equal(npout, npout2, decimal=decimal)


# Different operand fusions inside expressions
@pytest.mark.parametrize("expr", [
    "x + y",
    "(x + y) + z",
    "(x + y) * (x + z)",
    "(x + y + z) * (x + z)",
    "(x + y - z) * (x + y + t)",
    "(x - z + t) * (x + y - z)",
    "(x - z + t) * (z + t - x)",
    "(x - z + t + y) * (t - y + z - x)",
])
def test_expr_fusion(expr):
    shape = [200, 300]
    chunkshape = [40, 50]
    bshape = [20, 20]
    storage = ia.StorageProperties(chunkshape=chunkshape, blockshape=bshape)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(ia.dtshape(shape, dtype), .1, .9, storage=storage)
        y = ia.linspace(ia.dtshape(shape, dtype), 0., 1., storage=storage)
        z = ia.linspace(ia.dtshape(shape, dtype), 0., 2., storage=storage)
        t = ia.linspace(ia.dtshape(shape, dtype), 0., 3., storage=storage)

        # NumPy computation
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)
        npz = ia.iarray2numpy(z)
        npt = ia.iarray2numpy(t)
        npout = eval("%s" % expr, {"np": np, "x": npx, "y": npy, "z": npz, "t": npt})

        # High-level ironarray eval
        lazy_expr = eval(expr, {"x": x, "y": y, "z": z, "t": t})
        iout2 = lazy_expr.eval(dtype=dtype)
        npout2 = ia.iarray2numpy(iout2)

        decimal = 6 if dtype is np.float32 else 7
        np.testing.assert_almost_equal(npout, npout2, decimal=decimal)
