import pytest
import iarray as ia
import numpy as np


# Expression
@pytest.mark.parametrize("eval_flags, shape, pshape, dtype, expression", [
     ("iterblosc2", [1000], [110], np.float64, "x"),
     ("iterblosc", [1000], [100], np.float64, "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)"),
     ("auto", [1000], [100], np.float64, "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)"),
     ("iterchunk", [1000], [123], np.float32, "(abs(-x) - 1.35) * ceil(x) * floor(x - 8.5)"),
     # ("iterblosc2", [100, 100], [23, 32], np.float64, "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)"),  //TODO: Fix this
     ("iterchunk", [100, 100, 55], [10, 5, 10], np.float64, "asin(x) + (acos(x) - 1.35) - atan(x + .2)"),
     ("auto", [1000], None, np.float64, "exp(x) + (log(x) - 1.35) - log10(x + .2)"),
     ("iterchunk", [1000], None, np.float32, "sqrt(x) + atan2(x, x) + pow(x, x)"),
     ("auto", [100, 100], None, np.float64, "(x - cos(1)) * 2"),
     ("iterchunk", [8, 6, 7, 4, 5], None, np.float32, "(x - cos(y)) * (sin(x) + y) + 2 * x + y"),
     #("iterblosc", [8, 6, 7, 4, 5], [4, 3, 3, 4, 5], np.float32, "(x - cos(y)) * (sin(x) + y) + 2 * x + y"),
])
def test_expression(eval_flags, shape, pshape, dtype, expression):
    # The ranges below are important for not overflowing operations
    if pshape is None:
        storage = ia.StorageProperties(backend = "plainbuffer")
    else:
        storage = ia.StorageProperties(backend="blosc", enforce_frame=False, filename=None)

    x = ia.linspace(ia.dtshape(shape, pshape, dtype), 2.1, .2, storage=storage)
    y = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 1, storage=storage)
    npx = ia.iarray2numpy(x)
    npy = ia.iarray2numpy(y)

    expr = ia.Expr(eval_flags=eval_flags)
    expr.bind("x", x)
    expr.bind("y", y)
    expr.out_properties(ia.dtshape(shape, pshape, dtype), storage=storage)

    expr.compile(expression)

    iout = expr.eval()
    
    npout = ia.iarray2numpy(iout)
    npout2 = ia.Parser().parse(expression).evaluate({"x": npx, "y": npy})

    rtol = 1e-6 if dtype == np.dtype(np.float32) else 1e-14

    np.testing.assert_allclose(npout, npout2, rtol=rtol)


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
    #("negative(x)", "negate(x)"),
    ("power(x, y)", "pow(x, y)"),
    ("sin(x)", "sin(x)"),
    ("sinh(x)", "sinh(x)"),
    ("sqrt(x)", "sqrt(x)"),
    ("tan(x)", "tan(x)"),
    ("tanh(x)", "tanh(x)"),
])
def test_ufuncs(ufunc, ia_expr):
    shape = [20, 30]
    pshape = [2, 3]

    if pshape is None:
        storage = ia.StorageProperties(backend = "plainbuffer")
    else:
        storage = ia.StorageProperties(backend="blosc", enforce_frame=False, filename=None)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(ia.dtshape(shape, pshape, dtype), .1, .9, storage=storage)
        y = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 1, storage=storage)
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)

        # Low-level ironarray eval
        expr = ia.Expr(eval_flags="iterchunk")
        expr.bind("x", x)
        expr.bind("y", y)
        expr.out_properties(ia.dtshape(shape, pshape, dtype), storage=storage)
        expr.compile(ia_expr)
        iout = expr.eval()
        npout = ia.iarray2numpy(iout)

        decimal = 6 if dtype is np.float32 else 7

        # High-level ironarray eval
        lazy_expr = eval("ia." + ufunc, {"ia": ia, "x": x, "y": y})
        iout2 = lazy_expr.eval(eval_flags="iterchunk", pshape=pshape, dtype=dtype)
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
            iout2 = lazy_expr.eval(eval_flags="iterchunk", pshape=pshape, dtype=dtype)
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
    #"negative",
    "power",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
])
def test_expr_ufuncs(ufunc):
    shape = [20, 30]
    pshape = [4, 5]
    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(ia.dtshape(shape, pshape, dtype), .1, .9)
        y = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 1)

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
        iout2 = lazy_expr.eval(eval_flags="iterchunk", pshape=pshape, dtype=dtype)
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
    shape = [20, 30]
    pshape = [4, 5]
    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(ia.dtshape(shape, pshape, dtype), .1, .9)
        y = ia.linspace(ia.dtshape(shape, pshape, dtype), 0., 1.)
        z = ia.linspace(ia.dtshape(shape, pshape, dtype), 0., 2.)
        t = ia.linspace(ia.dtshape(shape, pshape, dtype), 0., 3.)

        # NumPy computation
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)
        npz = ia.iarray2numpy(z)
        npt = ia.iarray2numpy(t)
        npout = eval("%s" % expr, {"np": np, "x": npx, "y": npy, "z": npz, "t": npt})

        # High-level ironarray eval
        lazy_expr = eval(expr, {"x": x, "y": y, "z": z, "t": t})
        iout2 = lazy_expr.eval(eval_flags="iterchunk", pshape=pshape, dtype=dtype)
        npout2 = ia.iarray2numpy(iout2)

        decimal = 6 if dtype is np.float32 else 7
        np.testing.assert_almost_equal(npout, npout2, decimal=decimal)
