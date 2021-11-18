import pytest
import iarray as ia
import numpy
import numpy as np


# Expression
@pytest.mark.parametrize(
    "method, shape, chunks, blocks, dtype, expression, xcontiguous, xurlpath, ycontiguous, yurlpath",
    [
        (
            ia.Eval.ITERBLOSC,
            [100, 100],
            [23, 32],
            [10, 10],
            np.float64,
            "cos(x)",
            True,
            "test_expression_xcontiguous.iarr",
            True,
            "test_expression_ycontiguous.iarr",
        ),  # TODO: fix this
        (
            ia.Eval.ITERBLOSC,
            [100, 100],
            [10, 99],
            [4, 12],
            np.float64,
            "x",
            False,
            None,
            False,
            None,
        ),
        (ia.Eval.ITERBLOSC, [1000], [110], [55], np.float32, "x", True, None, True, None),
        (
            ia.Eval.ITERBLOSC,
            [1000],
            [100],
            [30],
            np.float64,
            "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)",
            False,
            "test_expression_xsparse.iarr",
            False,
            "test_expression_ysparse.iarr",
        ),
        (
            ia.Eval.AUTO,
            [1000],
            [100],
            [25],
            np.float64,
            "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)",
            True,
            None,
            False,
            "test_expression_ysparse.iarr",
        ),
        (
            ia.Eval.ITERCHUNK,
            [1000],
            [367],
            [77],
            np.float32,
            "(abs(-x) - 1.35) * ceil(x) * floor(x - 8.5)",
            False,
            None,
            True,
            None,
        ),
        (
            ia.Eval.ITERBLOSC,
            [100, 100, 100],
            [25, 25, 33],
            [12, 16, 8],
            np.float64,
            "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)",
            True,
            "test_expression_xcontiguous.iarr",
            True,
            None,
        ),
        (
            ia.Eval.ITERBLOSC,
            [223],
            [100],
            [30],
            np.float64,
            "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            "test_expression_ysparse.iarr",
        ),
        (
            ia.Eval.ITERCHUNK,
            [100, 100, 55],
            [10, 5, 10],
            [3, 4, 3],
            np.float64,
            "asin(x) + (acos(x) - 1.35) - atan(x + .2)",
            False,
            None,
            True,
            "test_expression_ycontiguous.iarr",
        ),
        (
            ia.Eval.ITERCHUNK,
            [100, 100, 55],
            [10, 5, 10],
            [3, 4, 3],
            np.float64,
            "arcsin(x) + (arccos(x) - 1.35) - arctan(x + .2)",
            True,
            None,
            True,
            "test_expression_ycontiguous.iarr",
        ),  # check NumPy naming convention for ufuncs
        (
            ia.Eval.AUTO,
            [1000],
            [500],
            [100],
            np.float64,
            "exp(x) + (log(x) - 1.35) - log10(x + .2)",
            True,
            None,
            False,
            None,
        ),
        (
            ia.Eval.ITERCHUNK,
            [1000],
            [500],
            [200],
            np.float32,
            "sqrt(x) + atan2(x, x) + pow(x, x)",
            False,
            None,
            True,
            None,
        ),
        (
            ia.Eval.AUTO,
            [1000],
            [500],
            [250],
            np.float32,
            "sqrt(x) + arctan2(x, x) + power(x, x)",
            False,
            "test_expression_xsparse.iarr",
            True,
            "test_expression_ycontiguous.iarr",
        ),  # NumPy conventions
        (
            ia.Eval.AUTO,
            [100, 100],
            [50, 50],
            [25, 25],
            np.float64,
            "(x - cos(1)) * 2",
            True,
            None,
            False,
            "test_expression_ysparse.iarr",
        ),
        (
            ia.Eval.ITERCHUNK,
            [8, 6, 7, 4, 5],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            np.float32,
            "(x - cos(y)) * (sin(x) + y) + 2 * x + y",
            False,
            "test_expression_xsparse.iarr",
            True,
            None,
        ),
        (
            ia.Eval.ITERBLOSC,
            [17, 12, 15, 15, 8],
            [8, 6, 7, 4, 5],
            [4, 3, 3, 4, 5],
            np.float64,
            "(x - cos(y)) * (sin(x) + y) + 2 * x + y",
            False,
            None,
            False,
            "test_expression_ysparse.iarr",
        ),
        (
            ia.Eval.ITERBLOSC,
            [17, 12, 15, 15, 8],
            [8, 6, 7, 4, 5],
            [4, 3, 3, 4, 5],
            np.float64,
            "(x - cos(0.5)) * (sin(.1) + y) + 2 * x + y",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            "test_expression_ysparse.iarr",
        ),
    ],
)
def test_expression(
    method, shape, chunks, blocks, dtype, expression, xcontiguous, xurlpath, ycontiguous, yurlpath
):
    # The ranges below are important for not overflowing operations
    ia.remove_urlpath(xurlpath)
    ia.remove_urlpath(yurlpath)
    ia.remove_urlpath("test_expression_zarray.iarr")

    xstore = ia.Store(chunks=chunks, blocks=blocks, contiguous=xcontiguous, urlpath=xurlpath)
    ystore = ia.Store(chunks=chunks, blocks=blocks, contiguous=ycontiguous, urlpath=yurlpath)
    zstore = ia.Store(
        chunks=chunks, blocks=blocks, contiguous=xcontiguous, urlpath="test_expression_zarray.iarr"
    )

    x = ia.linspace(shape, 0.1, 0.2, dtype=dtype, store=xstore)
    y = ia.linspace(shape, 0, 1, dtype=dtype, store=ystore)
    npx = ia.iarray2numpy(x)
    npy = ia.iarray2numpy(y)

    expr = ia.expr_from_string(expression, {"x": x, "y": y}, store=zstore, eval_method=method)
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

    ia.remove_urlpath(xstore.urlpath)
    ia.remove_urlpath(ystore.urlpath)
    ia.remove_urlpath(zstore.urlpath)


# ufuncs
@pytest.mark.parametrize(
    "ufunc, ia_expr, xcontiguous, xurlpath, ycontiguous, yurlpath",
    [
        ("abs(x)", "abs(x)", True, None, True, None),
        ("arccos(x)", "acos(x)", False, None, False, None),
        ("arcsin(x)", "asin(x)", True, "test_expression_xcontiguous.iarr", True, None),
        (
            "arctan(x)",
            "atan(x)",
            False,
            "test_expression_xsparse.iarr",
            True,
            "test_expression_ycontiguous.iarr",
        ),
        (
            "arctan2(x, y)",
            "atan2(x, y)",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            "test_expression_ysparse.iarr",
        ),
        ("ceil(x)", "ceil(x)", False, None, True, None),
        ("cos(x)", "cos(x)", True, None, False, None),
        ("cosh(x)", "cosh(x)", True, "test_expression_xcontiguous.iarr", False, None),
        (
            "exp(x)",
            "exp(x)",
            False,
            "test_expression_xsparse.iarr",
            True,
            "test_expression_ycontiguous.iarr",
        ),
        ("floor(x)", "floor(x)", False, None, False, None),
        ("log(x)", "log(x)", True, None, False, None),
        ("log10(x)", "log10(x)", True, "test_expression_xcontiguous.iarr", False, None),
        # ("negative(x)", "negate(x)"),
        (
            "power(x, y)",
            "pow(x, y)",
            False,
            "test_expression_xsparse.iarr",
            False,
            "test_expression_ysparse.iarr",
        ),
        ("sin(x)", "sin(x)", True, None, True, None),
        ("sinh(x)", "sinh(x)", False, None, False, None),
        ("sqrt(x)", "sqrt(x)", True, "test_expression_xcontiugous.iarr", True, None),
        ("tan(x)", "tan(x)", True, None, True, "test_expression_ycontiguous.iarr"),
        ("tanh(x)", "tanh(x)", False, None, False, "test_expression_ysparse.iarr"),
    ],
)
def test_ufuncs(ufunc, ia_expr, xcontiguous, xurlpath, ycontiguous, yurlpath):
    shape = [200, 300]
    chunks = [40, 40]
    bshape = [10, 17]

    xstore = ia.Store(chunks=chunks, blocks=bshape, contiguous=xcontiguous, urlpath=xurlpath)
    ystore = ia.Store(chunks=chunks, blocks=bshape, contiguous=ycontiguous, urlpath=yurlpath)
    if yurlpath != None:
        zstore = ia.Store(
            chunks=chunks,
            blocks=bshape,
            contiguous=ycontiguous,
            urlpath="test_expression_res.iarr",
        )
    else:
        zstore = ystore

    ia.remove_urlpath(xstore.urlpath)
    ia.remove_urlpath(ystore.urlpath)
    ia.remove_urlpath(zstore.urlpath)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, store=xstore)
        y = ia.linspace(shape, 0, 1, dtype=dtype, store=ystore)
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)

        expr = ia.expr_from_string(ia_expr, {"x": x, "y": y}, store=zstore)
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

        ia.remove_urlpath(xstore.urlpath)
        ia.remove_urlpath(ystore.urlpath)
        ia.remove_urlpath(zstore.urlpath)


# ufuncs inside of expressions
@pytest.mark.parametrize(
    "ufunc, xcontiguous, xurlpath, ycontiguous, yurlpath",
    [
        ("abs", True, None, True, None),
        ("arccos", False, "test_expression_xsparse.iarr", True, None),
        ("arcsin", False, None, True, "test_expression_ycontiguous.iarr"),
        ("arctan", False, None, False, None),
        ("arctan2", True, None, True, "test_expression_ycontiguous.iarr"),
        ("ceil", False, None, False, "test_expression_ysparse.iarr"),
        (
            "cos",
            True,
            "test_expression_xcontiguous.iarr",
            True,
            "test_expression_ycontiguous.iarr",
        ),
        ("cosh", True, None, False, None),
        ("exp", False, None, True, None),
        ("floor", False, "test_expression_xsparse.iarr", False, "test_expression_ysparse.iarr"),
        ("log", False, "test_expression_xsparse.iarr", True, "test_expression_ycontiguous.iarr"),
        ("log10", True, "test_expression_xcontiguous.iarr", False, "test_expression_ysparse.iarr"),
        # "negative",
        ("power", True, "test_expression_xcontiguous.iarr", True, None),
        ("sin", True, "test_expression_xcontiguous.iarr", False, None),
        ("sinh", True, None, False, "test_expression_ysparse.iarr"),
        ("sqrt", False, None, True, None),
        ("tan", False, "test_expression_xsparse.iarr", False, "test_expression_ysparse"),
        ("tanh", True, None, True, None),
    ],
)
def test_expr_ufuncs(ufunc, xcontiguous, xurlpath, ycontiguous, yurlpath):
    shape = [200, 300]
    cshape = [40, 50]
    bshape = [20, 20]

    xstore = ia.Store(chunks=cshape, blocks=bshape, contiguous=xcontiguous, urlpath=xurlpath)
    x32store = ia.Store(chunks=cshape, blocks=bshape, contiguous=xcontiguous, urlpath="x32urlpath")
    if xstore.urlpath is None:
        x32store.urlpath = None
    ystore = ia.Store(chunks=cshape, blocks=bshape, contiguous=ycontiguous, urlpath=yurlpath)
    y32store = ia.Store(chunks=cshape, blocks=bshape, contiguous=ycontiguous, urlpath="y32urlpath")
    if ystore.urlpath is None:
        y32store.urlpath = None

    ia.remove_urlpath(xstore.urlpath)
    ia.remove_urlpath(ystore.urlpath)
    ia.remove_urlpath(x32store.urlpath)
    ia.remove_urlpath(y32store.urlpath)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        if dtype == np.float32:
            x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, store=x32store)
            y = ia.linspace(shape, 0.5, 1, dtype=dtype, store=y32store)
        else:
            x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, store=xstore)
            y = ia.linspace(shape, 0.5, 1, dtype=dtype, store=ystore)

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

    ia.remove_urlpath(xstore.urlpath)
    ia.remove_urlpath(ystore.urlpath)
    ia.remove_urlpath(x32store.urlpath)
    ia.remove_urlpath(y32store.urlpath)


# Different operand fusions inside expressions
@pytest.mark.parametrize(
    "expr, np_expr, xcontiguous, xurlpath, ycontiguous, yurlpath, zcontiguous, zurlpath, tcontiguous, turlpath",
    [
        (
            "x + y",
            "x + y",
            True,
            "test_expression_xcontiguous.iarr",
            True,
            None,
            True,
            None,
            True,
            None,
        ),
        (
            "(x + y) + z",
            "(x + y) + z",
            False,
            "test_expression_xsparse.iarr",
            True,
            "test_expression_ycontiguous.iarr",
            True,
            "test_expression_zcontiguous.iarr",
            False,
            None,
        ),
        (
            "(x + y) * (x + z)",
            "(x + y) * (x + z)",
            False,
            None,
            False,
            "test_expression_ysparse.iarr",
            False,
            None,
            False,
            None,
        ),
        (
            "(x + y + z) * (x + z)",
            "(x + y + z) * (x + z)",
            True,
            None,
            False,
            None,
            False,
            None,
            False,
            None,
        ),
        (
            "(x + y - z) * (x + y + t)",
            "(x + y - z) * (x + y + t)",
            False,
            None,
            False,
            None,
            False,
            None,
            False,
            None,
        ),
        (
            "(x - z + t) * (x + y - z)",
            "(x - z + t) * (x + y - z)",
            False,
            None,
            False,
            "test_expression_ysparse.iarr",
            False,
            "test_expression_zsparse.iarr",
            True,
            None,
        ),
        (
            "(x - z + t) * (z + t - x)",
            "(x - z + t) * (z + t - x)",
            True,
            "test_expression_xcontiguous.iarr",
            True,
            "test_expression_ycontiguous.iarr",
            True,
            "test_expression_zcontiguous.iarr",
            True,
            "test_expression_tcontiguous.iarr",
        ),
        (
            "(x - z + t + y) * (t - y + z - x)",
            "(x - z + t + y) * (t - y + z - x)",
            False,
            "test_expression_xsparse.iarr",
            False,
            "test_expression_ysparse.iarr",
            False,
            "test_expression_zsparse.iarr",
            False,
            "test_expression_tsparse.iarr",
        ),
        (
            "(x - z + t + y) * (t - y + z - x)",
            "(x - z + t + y) * (t - y + z - x)",
            True,
            None,
            True,
            None,
            True,
            None,
            True,
            None,
        ),
        # transcendental functions
        (
            "x.cos() + y",
            "np.cos(x) + y",
            True,
            "test_expression_xcontiguous.iarr",
            True,
            "test_expression_ycontiguous.iarr",
            True,
            None,
            True,
            None,
        ),
        ("ia.cos(x) + y", "np.cos(x) + y", False, None, False, None, False, None, False, None),
        (
            "x.sin() * x.sin() + y.cos()",
            "np.sin(x) * np.sin(x) + np.cos(y)",
            False,
            "test_expression_xsparse.iarr",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            None,
            False,
            None,
        ),
        (
            "x.tan() * (y.sin() * y.sin() + z.cos()) + (t.sqrt() * 2)",
            "np.tan(x) * (np.sin(y) * np.sin(y) + np.cos(z)) + (np.sqrt(t) * 2)",
            True,
            None,
            False,
            "test_expression_ysparse.iarr",
            False,
            "test_expression_zsparse.iarr",
            True,
            "test_expression_tsparse.iarr",
        ),
        # Use another order than before (precision needs to be relaxed a bit)
        (
            "t.tan() * (x.sin() * x.sin() + y.cos()) + (z.sqrt() * 2)",
            "np.tan(t) * (np.sin(x) * np.sin(x) + np.cos(y)) + (np.sqrt(z) * 2)",
            False,
            None,
            True,
            None,
            False,
            None,
            True,
            None,
        ),
    ],
)
def test_expr_fusion(
    expr,
    np_expr,
    xcontiguous,
    xurlpath,
    ycontiguous,
    yurlpath,
    zcontiguous,
    zurlpath,
    tcontiguous,
    turlpath,
):
    shape = [200, 300]
    chunks = [40, 50]
    bshape = [20, 20]

    xstore = ia.Store(chunks=chunks, blocks=bshape, contiguous=xcontiguous, urlpath=xurlpath)
    x32store = ia.Store(chunks=chunks, blocks=bshape, contiguous=xcontiguous, urlpath="x32urlpath")
    ystore = ia.Store(chunks=chunks, blocks=bshape, contiguous=ycontiguous, urlpath=yurlpath)
    y32store = ia.Store(chunks=chunks, blocks=bshape, contiguous=ycontiguous, urlpath="y32urlpath")
    zstore = ia.Store(chunks=chunks, blocks=bshape, contiguous=zcontiguous, urlpath=zurlpath)
    z32store = ia.Store(chunks=chunks, blocks=bshape, contiguous=zcontiguous, urlpath="z32urlpath")
    tstore = ia.Store(chunks=chunks, blocks=bshape, contiguous=tcontiguous, urlpath=turlpath)
    t32store = ia.Store(chunks=chunks, blocks=bshape, contiguous=tcontiguous, urlpath="t32urlpath")
    if xstore.urlpath is None:
        x32store.urlpath = None
    if ystore.urlpath is None:
        y32store.urlpath = None
    if zstore.urlpath is None:
        z32store.urlpath = None
    if tstore.urlpath is None:
        t32store.urlpath = None

    ia.remove_urlpath(xstore.urlpath)
    ia.remove_urlpath(ystore.urlpath)
    ia.remove_urlpath(x32store.urlpath)
    ia.remove_urlpath(y32store.urlpath)
    ia.remove_urlpath(zstore.urlpath)
    ia.remove_urlpath(tstore.urlpath)
    ia.remove_urlpath(z32store.urlpath)
    ia.remove_urlpath(t32store.urlpath)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        if dtype == np.float32:
            x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, store=x32store)
            y = ia.linspace(shape, 0.5, 1, dtype=dtype, store=y32store)
            z = ia.linspace(shape, 0.1, 0.9, dtype=dtype, store=z32store)
            t = ia.linspace(shape, 0.5, 1, dtype=dtype, store=t32store)
        else:
            x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, store=xstore)
            y = ia.linspace(shape, 0.5, 1, dtype=dtype, store=ystore)
            z = ia.linspace(shape, 1.0, 2.0, dtype=dtype, store=zstore)
            t = ia.linspace(shape, 1.5, 3.0, dtype=dtype, store=tstore)

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

    ia.remove_urlpath(xstore.urlpath)
    ia.remove_urlpath(ystore.urlpath)
    ia.remove_urlpath(x32store.urlpath)
    ia.remove_urlpath(y32store.urlpath)
    ia.remove_urlpath(zstore.urlpath)
    ia.remove_urlpath(tstore.urlpath)
    ia.remove_urlpath(z32store.urlpath)
    ia.remove_urlpath(t32store.urlpath)
