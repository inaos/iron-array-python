import pytest
import iarray as ia
import numpy
import numpy as np


# Expression
@pytest.mark.parametrize(
    "method, shape, chunks, blocks, dtype, expression",
    [
        (
            ia.Eval.ITERBLOSC,
            [100, 100],
            [23, 32],
            [10, 10],
            np.float64,
            "cos(x)",
        ),  # TODO: fix this
        (ia.Eval.ITERBLOSC, [100, 100], [10, 99], [4, 12], np.float64, "x"),
        (ia.Eval.ITERBLOSC, [1000], [110], [55], np.float32, "x"),
        (
            ia.Eval.ITERBLOSC,
            [1000],
            [100],
            [30],
            np.float64,
            "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)",
        ),
        (
            ia.Eval.AUTO,
            [1000],
            [100],
            [25],
            np.float64,
            "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)",
        ),
        (
            ia.Eval.ITERCHUNK,
            [1000],
            [367],
            [77],
            np.float32,
            "(abs(-x) - 1.35) * ceil(x) * floor(x - 8.5)",
        ),
        (
            ia.Eval.ITERBLOSC,
            [100, 100, 100],
            [25, 25, 33],
            [12, 16, 8],
            np.float64,
            "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)",
        ),
        (
            ia.Eval.ITERBLOSC,
            [223],
            [100],
            [30],
            np.float64,
            "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)",
        ),
        (
            ia.Eval.ITERCHUNK,
            [100, 100, 55],
            [10, 5, 10],
            [3, 4, 3],
            np.float64,
            "asin(x) + (acos(x) - 1.35) - atan(x + .2)",
        ),
        (
            ia.Eval.ITERCHUNK,
            [100, 100, 55],
            [10, 5, 10],
            [3, 4, 3],
            np.float64,
            "arcsin(x) + (arccos(x) - 1.35) - arctan(x + .2)",
        ),  # check NumPy naming convention for ufuncs
        (ia.Eval.AUTO, [1000], None, None, np.float64, "exp(x) + (log(x) - 1.35) - log10(x + .2)"),
        (ia.Eval.ITERCHUNK, [1000], None, None, np.float32, "sqrt(x) + atan2(x, x) + pow(x, x)"),
        (
            ia.Eval.AUTO,
            [1000],
            None,
            None,
            np.float32,
            "sqrt(x) + arctan2(x, x) + power(x, x)",
        ),  # NumPy conventions
        (ia.Eval.AUTO, [100, 100], None, None, np.float64, "(x - cos(1)) * 2"),
        (
            ia.Eval.ITERCHUNK,
            [8, 6, 7, 4, 5],
            None,
            None,
            np.float32,
            "(x - cos(y)) * (sin(x) + y) + 2 * x + y",
        ),
        (
            ia.Eval.ITERBLOSC,
            [17, 12, 15, 15, 8],
            [8, 6, 7, 4, 5],
            [4, 3, 3, 4, 5],
            np.float64,
            "(x - cos(y)) * (sin(x) + y) + 2 * x + y",
        ),
        (
            ia.Eval.ITERBLOSC,
            [17, 12, 15, 15, 8],
            [8, 6, 7, 4, 5],
            [4, 3, 3, 4, 5],
            np.float64,
            "(x - cos(0.5)) * (sin(.1) + y) + 2 * x + y",
        ),
    ],
)
def test_expression(method, shape, chunks, blocks, dtype, expression):
    # The ranges below are important for not overflowing operations
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks=chunks, blocks=blocks)

    x = ia.linspace(shape, 0.1, 0.2, dtype=dtype, store=store)
    y = ia.linspace(shape, 0, 1, dtype=dtype, store=store)
    npx = ia.iarray2numpy(x)
    npy = ia.iarray2numpy(y)

    expr = ia.expr_from_string(expression, {"x": x, "y": y}, store=store, eval_method=method)
    iout = expr.eval()
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
    for ufunc in ia.UFUNC_LIST:
        if ufunc in expression:
            idx = expression.find(ufunc)
            # Prevent replacing an ufunc with np.ufunc twice (not terribly solid, but else, test will crash)
            if "np." not in expression[idx - len("np.arc") : idx]:
                expression = expression.replace(ufunc + "(", "np." + ufunc + "(")
    npout2 = eval(expression, {"x": npx, "y": npy, "np": numpy})

    tol = 1e-6 if dtype is np.float32 else 1e-14
    np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)


# ufuncs
@pytest.mark.parametrize(
    "ufunc, ia_expr",
    [
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
    ],
)
def test_ufuncs(ufunc, ia_expr):
    shape = [200, 300]
    chunks = [40, 40]
    bshape = [10, 17]

    store = ia.Store(chunks=chunks, blocks=bshape)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, store=store)
        y = ia.linspace(shape, 0, 1, dtype=dtype, store=store)
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)

        expr = ia.expr_from_string(ia_expr, {"x": x, "y": y}, store=store)
        iout = expr.eval()
        npout = ia.iarray2numpy(iout)

        tol = 1e-5 if dtype is np.float32 else 1e-13

        # Lazy expression eval
        lazy_expr = eval("ia." + ufunc, {"ia": ia, "x": x, "y": y})
        iout2 = lazy_expr.eval()
        npout2 = ia.iarray2numpy(iout2)
        np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)

        # Lazy expression eval, but via numpy ufunc machinery
        # TODO: the next ufuncs still have some problems with the numpy machinery (bug?)
        # abs(x) : TypeError: bad operand type for abs(): 'IArray'
        # ceil(x) : TypeError: must be real number, not IArray
        # floor(x): TypeError: must be real number, not IArray
        # negative(x) : TypeError: bad operand type for unary -: 'IArray'
        # power(x,y) : TypeError: unsupported operand type(s) for ** or pow(): 'IArray' and 'IArray'
        if ufunc not in ("abs(x)", "ceil(x)", "floor(x)", "negative(x)", "power(x, y)"):
            lazy_expr = eval("np." + ufunc, {"np": np, "x": x, "y": y})
            iout2 = lazy_expr.eval()
            npout2 = ia.iarray2numpy(iout2)
            np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)

        npout2 = eval("np." + ufunc, {"np": np, "x": npx, "y": npy})  # pure numpy
        np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)


# ufuncs inside of expressions
@pytest.mark.parametrize(
    "ufunc",
    [
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
    ],
)
def test_expr_ufuncs(ufunc):
    shape = [200, 300]
    cshape = [40, 50]
    bshape = [20, 20]
    store = ia.Store(chunks=cshape, blocks=bshape)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, store=store)
        y = ia.linspace(shape, 0.5, 1, dtype=dtype, store=store)

        # NumPy computation
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)
        if ufunc in ("arctan2", "power"):
            npout = eval("1 + 2 * np.%s(x, y)" % ufunc, {"np": np, "x": npx, "y": npy})
        else:
            npout = eval("1 + 2 * np.%s(x)" % ufunc, {"np": np, "x": npx})

        # Lazy expression eval
        if ufunc in ("arctan2", "power"):
            lazy_expr = eval("1 + 2* x.%s(y)" % ufunc, {"x": x, "y": y})
        else:
            lazy_expr = eval("1 + 2 * x.%s()" % ufunc, {"x": x})
        iout2 = lazy_expr.eval()
        npout2 = ia.iarray2numpy(iout2)

        tol = 1e-5 if dtype is np.float32 else 1e-13
        np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)


# Different operand fusions inside expressions
@pytest.mark.parametrize(
    "expr, np_expr",
    [
        ("x + y", "x + y"),
        ("(x + y) + z", "(x + y) + z"),
        ("(x + y) * (x + z)", "(x + y) * (x + z)"),
        ("(x + y + z) * (x + z)", "(x + y + z) * (x + z)"),
        ("(x + y - z) * (x + y + t)", "(x + y - z) * (x + y + t)"),
        ("(x - z + t) * (x + y - z)", "(x - z + t) * (x + y - z)"),
        ("(x - z + t) * (z + t - x)", "(x - z + t) * (z + t - x)"),
        ("(x - z + t + y) * (t - y + z - x)", "(x - z + t + y) * (t - y + z - x)"),
        ("(x - z + t + y) * (t - y + z - x)", "(x - z + t + y) * (t - y + z - x)"),
        # transcendental functions
        ("x.cos() + y", "np.cos(x) + y"),
        ("ia.cos(x) + y", "np.cos(x) + y"),
        ("x.sin() * x.sin() + y.cos()", "np.sin(x) * np.sin(x) + np.cos(y)"),
        (
            "x.tan() * (y.sin() * y.sin() + z.cos()) + (t.sqrt() * 2)",
            "np.tan(x) * (np.sin(y) * np.sin(y) + np.cos(z)) + (np.sqrt(t) * 2)",
        ),
        # Use another order than before (precision needs to be relaxed a bit)
        (
            "t.tan() * (x.sin() * x.sin() + y.cos()) + (z.sqrt() * 2)",
            "np.tan(t) * (np.sin(x) * np.sin(x) + np.cos(y)) + (np.sqrt(z) * 2)",
        ),
    ],
)
def test_expr_fusion(expr, np_expr):
    shape = [200, 300]
    chunks = [40, 50]
    bshape = [20, 20]
    store = ia.Store(chunks=chunks, blocks=bshape)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, store=store)
        y = ia.linspace(shape, 0.5, 1.0, dtype=dtype, store=store)
        z = ia.linspace(shape, 1.0, 2.0, dtype=dtype, store=store)
        t = ia.linspace(shape, 1.5, 3.0, dtype=dtype, store=store)

        # NumPy computation
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)
        npz = ia.iarray2numpy(z)
        npt = ia.iarray2numpy(t)
        npout = eval("%s" % np_expr, {"np": np, "x": npx, "y": npy, "z": npz, "t": npt})

        # High-level ironarray eval
        lazy_expr = eval(expr, {"ia": ia, "x": x, "y": y, "z": z, "t": t})
        iout2 = lazy_expr.eval()
        npout2 = ia.iarray2numpy(iout2)

        tol = 1e-6 if dtype is np.float32 else 1e-14
        np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)
