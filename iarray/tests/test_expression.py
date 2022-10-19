import pytest
import iarray as ia
import numpy
import numpy as np


# Expression
@pytest.mark.parametrize(
    "method, shape, chunks, blocks, dtype, np_dtype, expression, xcontiguous, xurlpath, ycontiguous, yurlpath",
    [
        (
            ia.Eval.ITERBLOSC,
            [100, 100],
            [23, 32],
            [10, 10],
            np.float64,
            None,
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
            np.int64,
            "M8[Y]",
            "x",
            False,
            None,
            False,
            None,
        ),
        (ia.Eval.ITERBLOSC, [100], [20], [5], np.int64, "M8[M]", "x-y", True, None, True, None),
        (ia.Eval.ITERBLOSC, [1000], [110], [55], np.float32, None, "y", True, None, True, None),
        (
            ia.Eval.ITERBLOSC,
            [1000],
            [100],
            [30],
            np.float64,
            None,
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
            None,
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
            None,
            "(abs(-x) - 1.35) * ceil(x) * floor(x - 8.5)",
            False,
            None,
            True,
            None,
        ),
        pytest.param(
            ia.Eval.ITERBLOSC,
            [100, 100, 100],
            [25, 25, 33],
            [12, 16, 8],
            np.float64,
            None,
            "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)",
            True,
            "test_expression_xcontiguous.iarr",
            True,
            None,
            marks=pytest.mark.heavy,
        ),
        pytest.param(
            ia.Eval.ITERBLOSC,
            [223],
            [100],
            [30],
            np.float64,
            None,
            "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            "test_expression_ysparse.iarr",
            marks=pytest.mark.heavy,
        ),
        pytest.param(
            ia.Eval.ITERCHUNK,
            [40, 50, 55],
            [10, 5, 10],
            [3, 4, 3],
            np.float64,
            None,
            "asin(x) + (acos(x) - 1.35) - atan(x + .2)",
            False,
            None,
            True,
            "test_expression_ycontiguous.iarr",
            marks=pytest.mark.heavy,
        ),
        (
            ia.Eval.ITERCHUNK,
            [50, 10, 55],
            [10, 5, 10],
            [3, 4, 3],
            np.float64,
            None,
            "asin(x) + (acos(x) - 1.35) - atan(x + .2)",
            False,
            None,
            True,
            "test_expression_ycontiguous.iarr",
        ),
        (
            ia.Eval.ITERCHUNK,
            [100, 55],
            [10, 10],
            [3, 3],
            np.float64,
            None,
            "arcsin(x) + (arccos(x) - 1.35) - arctan(x + .2)",
            True,
            None,
            True,
            "test_expression_ycontiguous.iarr",
        ),
        pytest.param(
            ia.Eval.ITERCHUNK,
            [100, 50, 55],
            [10, 5, 10],
            [3, 4, 3],
            np.float64,
            None,
            "arcsin(x) + (arccos(x) - 1.35) - arctan(x + .2)",
            True,
            None,
            True,
            "test_expression_ycontiguous.iarr",
            marks=pytest.mark.heavy,
        ),  # check NumPy naming convention for ufuncs
        (
            ia.Eval.AUTO,
            [1000],
            [500],
            [100],
            np.float64,
            None,
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
            None,
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
            None,
            "sqrt(x) + atan2(x, x) + pow(x, x)",
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
            None,
            "(x - cos(1)) * 2",
            True,
            None,
            False,
            "test_expression_ysparse.iarr",
        ),
        pytest.param(
            ia.Eval.ITERCHUNK,
            [8, 6, 7, 4],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            np.float32,
            None,
            "(x - cos(y)) * (sin(x) + y) + 2 * x + y",
            False,
            "test_expression_xsparse.iarr",
            True,
            None,
            marks=pytest.mark.heavy,
        ),
        (
            ia.Eval.ITERBLOSC,
            [15, 8],
            [4, 5],
            [4, 3],
            np.float64,
            None,
            "(x - cos(y)) * (sin(x) + y) + 2 * x + y",
            False,
            None,
            False,
            "test_expression_ysparse.iarr",
        ),
        pytest.param(
            ia.Eval.ITERBLOSC,
            [17, 12, 15, 15, 8],
            [8, 6, 7, 4, 5],
            [4, 3, 3, 4, 5],
            np.float64,
            None,
            "(x - cos(y)) * (sin(x) + y) + 2 * x + y",
            False,
            None,
            False,
            "test_expression_ysparse.iarr",
            marks=pytest.mark.heavy,
        ),
        (
            ia.Eval.ITERBLOSC,
            [17, 12],
            [8, 6],
            [4, 3],
            np.float64,
            None,
            "(x - cos(0.5)) * (sin(.1) + y) + 2 * x + y",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            "test_expression_ysparse.iarr",
        ),
        pytest.param(
            ia.Eval.ITERBLOSC,
            [17, 12, 15, 15, 8],
            [8, 6, 7, 4, 5],
            [4, 3, 3, 4, 5],
            np.float64,
            None,
            "(x - cos(0.5)) * (sin(.1) + y) + 2 * x + y",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            "test_expression_ysparse.iarr",
            marks=pytest.mark.heavy,
        ),
        (
            ia.Eval.ITERBLOSC,
            [100, 100],
            [23, 32],
            [10, 10],
            np.float32,
            None,
            "2.71828**y / x",
            True,
            "test_expression_xcontiguous.iarr",
            True,
            "test_expression_ycontiguous.iarr",
        ),
    ],
)
def test_expression(
    method,
    shape,
    chunks,
    blocks,
    dtype,
    np_dtype,
    expression,
    xcontiguous,
    xurlpath,
    ycontiguous,
    yurlpath,
):
    # The ranges below are important for not overflowing operations
    ia.remove_urlpath(xurlpath)
    ia.remove_urlpath(yurlpath)
    ia.remove_urlpath("test_expression_zarray.iarr")

    x = ia.linspace(
        shape,
        0.1,
        0.2,
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=xcontiguous,
        urlpath=xurlpath,
    )
    y = ia.linspace(
        shape,
        0,
        1,
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=ycontiguous,
        urlpath=yurlpath,
    )
    npx = ia.iarray2numpy(x)
    npy = ia.iarray2numpy(y)

    out_dtype = (
        "m8[M]"
        if np_dtype is not None and np.dtype(np_dtype).str[1] == "M" and expression == "x-y"
        else np_dtype
    )
    expr = ia.expr_from_string(
        expression,
        {"x": x, "y": y},
        chunks=chunks,
        blocks=blocks,
        contiguous=xcontiguous,
        urlpath="test_expression_zarray.iarr",
        dtype=dtype,
        np_dtype=out_dtype,
        eval_method=method,
    )

    with pytest.raises(IOError):
        expr.cfg.mode = "r"
        expr.eval()
    with pytest.raises(IOError):
        expr.cfg.mode = "r+"
        expr.eval()
    expr.cfg.mode = "w-"
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
    for ufunc in ia.UNIVERSAL_MATH_FUNCS:
        if ufunc in expression:
            idx = expression.find(ufunc)
            # Prevent replacing an ufunc with np.ufunc twice (not terribly solid, but else, test will crash)
            if "np." not in expression[idx - len("np.arc") : idx]:
                expression = expression.replace(ufunc + "(", "np." + ufunc + "(")
    npout2 = eval(expression, {"x": npx, "y": npy, "np": numpy})

    if np_dtype is None:
        tol = 1e-6 if dtype is np.float32 else 1e-14
        np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)
    else:
        np.testing.assert_equal(npout, npout2)

    ia.remove_urlpath(x.cfg.urlpath)
    ia.remove_urlpath(y.cfg.urlpath)
    ia.remove_urlpath(iout.cfg.urlpath)


# ufuncs
@pytest.mark.parametrize(
    "ufunc, ia_expr, xcontiguous, xurlpath, ycontiguous, yurlpath",
    [
        ("abs(x)", "abs(x)", True, None, True, None),
        ("arccos(x)", "acos(x)", False, None, False, None),
        ("arcsin(x)", "asin(x)", True, None, True, None),
        (
            "arctan(x)",
            "atan(x)",
            False,
            "test_expression_xsparse.iarr",
            True,
            "test_expression_ycontiguous.iarr",
        ),
        pytest.param(
            "arctan2(x, y)",
            "atan2(x, y)",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            "test_expression_ysparse.iarr",
            marks=pytest.mark.heavy,
        ),
        ("ceil(x)", "ceil(x)", False, None, True, None),
        ("cos(x)", "cos(x)", True, None, False, None),
        ("cosh(x)", "cosh(x)", True, "test_expression_xcontiguous.iarr", False, None),
        pytest.param(
            "exp(x)",
            "exp(x)",
            False,
            "test_expression_xsparse.iarr",
            True,
            "test_expression_ycontiguous.iarr",
            marks=pytest.mark.heavy,
        ),
        ("floor(x)", "floor(x)", False, None, False, None),
        ("log(x)", "log(x)", True, None, False, None),
        ("log10(x)", "log10(x)", True, "test_expression_xcontiguous.iarr", False, None),
        ("negative(x)", "negative(x)", True, "test_expression_xcontiguous.iarr", False, None),
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
        pytest.param(
            "tan(x)",
            "tan(x)",
            True,
            None,
            True,
            "test_expression_ycontiguous.iarr",
            marks=pytest.mark.heavy,
        ),
        ("tanh(x)", "tanh(x)", False, None, False, "test_expression_ysparse.iarr"),
    ],
)
def test_ufuncs(ufunc, ia_expr, xcontiguous, xurlpath, ycontiguous, yurlpath):
    shape = [100, 150]
    chunks = [40, 40]
    bshape = [10, 17]

    ia.remove_urlpath(xurlpath)
    ia.remove_urlpath(yurlpath)
    ia.remove_urlpath("test_expression_res.iarr")

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(
            shape,
            0.1,
            0.9,
            dtype=dtype,
            chunks=chunks,
            blocks=bshape,
            contiguous=xcontiguous,
            urlpath=xurlpath,
        )
        y = ia.linspace(
            shape,
            0,
            1,
            dtype=dtype,
            chunks=chunks,
            blocks=bshape,
            contiguous=ycontiguous,
            urlpath=yurlpath,
        )
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)

        if y.cfg.urlpath is not None:
            expr = ia.expr_from_string(
                ia_expr,
                {"x": x, "y": y},
                chunks=chunks,
                blocks=bshape,
                contiguous=ycontiguous,
                urlpath="test_expression_res.iarr",
            )
        else:
            expr = ia.expr_from_string(ia_expr, {"x": x, "y": y}, y.cfg)
        iout = expr.eval()
        npout = ia.iarray2numpy(iout)

        tol = 1e-5 if dtype is np.float32 else 1e-13

        # Lazy expression eval
        lazy_expr = eval("ia." + ia_expr, {"ia": ia, "x": x, "y": y})
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
        # if ufunc not in ("abs(x)", "ceil(x)", "floor(x)", "negative(x)", "power(x, y)"):
        #     lazy_expr = eval("np." + ufunc, {"np": np, "x": x, "y": y})
        #     iout2 = lazy_expr.eval()
        #     npout2 = ia.iarray2numpy(iout2)
        # else:
        npout2 = eval("np." + ufunc, {"np": np, "x": x.data, "y": y.data})
        np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)

        npout2 = eval("np." + ufunc, {"np": np, "x": npx, "y": npy})  # pure numpy
        np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)

        ia.remove_urlpath(x.cfg.urlpath)
        ia.remove_urlpath(y.cfg.urlpath)
        ia.remove_urlpath(iout.cfg.urlpath)


# ufuncs inside of expressions
@pytest.mark.parametrize(
    "ufunc, xcontiguous, xurlpath, ycontiguous, yurlpath",
    [
        ("abs", True, None, True, None),
        ("acos", False, "test_expression_xsparse.iarr", True, None),
        ("asin", False, None, True, "test_expression_ycontiguous.iarr"),
        ("atan", False, None, False, None),
        ("atan2", True, None, True, "test_expression_ycontiguous.iarr"),
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
        (
            "negative",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            "test_expression_ysparse.iarr",
        ),
        ("pow", True, "test_expression_xcontiguous.iarr", True, None),
        ("sin", True, "test_expression_xcontiguous.iarr", False, None),
        ("sinh", True, None, False, "test_expression_ysparse.iarr"),
        ("sqrt", False, None, True, None),
        pytest.param(
            "tan",
            False,
            "test_expression_xsparse.iarr",
            False,
            "test_expression_ysparse",
            marks=pytest.mark.heavy,
        ),
        ("tanh", True, None, True, None),
    ],
)
def test_expr_ufuncs(ufunc, xcontiguous, xurlpath, ycontiguous, yurlpath):
    shape = [100, 150]
    cshape = [40, 50]
    bshape = [20, 20]

    xcfg = ia.Config(chunks=cshape, blocks=bshape, contiguous=xcontiguous, urlpath=xurlpath)
    x32cfg = ia.Config(chunks=cshape, blocks=bshape, contiguous=xcontiguous, urlpath="x32urlpath")
    if xcfg.urlpath is None:
        x32cfg.urlpath = None
    ycfg = ia.Config(chunks=cshape, blocks=bshape, contiguous=ycontiguous, urlpath=yurlpath)
    y32cfg = ia.Config(chunks=cshape, blocks=bshape, contiguous=ycontiguous, urlpath="y32urlpath")
    if ycfg.urlpath is None:
        y32cfg.urlpath = None

    ia.remove_urlpath(xcfg.urlpath)
    ia.remove_urlpath(ycfg.urlpath)
    ia.remove_urlpath(x32cfg.urlpath)
    ia.remove_urlpath(y32cfg.urlpath)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        if dtype == np.float32:
            x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, cfg=x32cfg)
            assert x.cfg.dtype == dtype
            y = ia.linspace(shape, 0.5, 1, dtype=dtype, cfg=y32cfg)
        else:
            x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, cfg=xcfg)
            assert x.cfg.dtype == dtype
            y = ia.linspace(shape, 0.5, 1, dtype=dtype, cfg=ycfg)

        # NumPy computation
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)
        np_ufunc = ufunc
        if np_ufunc in ["asin", "acos", "atan", "atan2"]:
            np_ufunc = "arc" + ufunc[1:]
        elif np_ufunc == "pow":
            np_ufunc = "power"
        if np_ufunc in ("arctan2", "power"):
            npout = eval("1 + 2 * np.%s(x, y)" % np_ufunc, {"np": np, "x": npx, "y": npy})
        else:
            npout = eval("1 + 2 * np.%s(x)" % np_ufunc, {"np": np, "x": npx})

        # Lazy expression eval
        if ufunc in ("atan2", "pow"):
            lazy_expr = eval("1 + 2* ia.%s(x, y)" % ufunc, {"ia": ia, "x": x, "y": y})
        else:
            lazy_expr = eval("1 + 2 * ia.%s(x)" % ufunc, {"ia": ia, "x": x})
        iout2 = lazy_expr.eval()
        npout2 = ia.iarray2numpy(iout2)

        tol = 1e-5 if dtype is np.float32 else 1e-13
        np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)

    ia.remove_urlpath(xcfg.urlpath)
    ia.remove_urlpath(ycfg.urlpath)
    ia.remove_urlpath(x32cfg.urlpath)
    ia.remove_urlpath(y32cfg.urlpath)


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
        pytest.param(
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
            marks=pytest.mark.heavy,
        ),
        pytest.param(
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
            marks=pytest.mark.heavy,
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
            "ia.cos(x) + y",
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
            "ia.sin(x) * ia.sin(x) + ia.cos(y)",
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
        pytest.param(
            "ia.tan(x) * (ia.sin(y) * ia.sin(y) + ia.cos(z)) + (ia.sqrt(t) * 2)",
            "np.tan(x) * (np.sin(y) * np.sin(y) + np.cos(z)) + (np.sqrt(t) * 2)",
            True,
            None,
            False,
            "test_expression_ysparse.iarr",
            False,
            "test_expression_zsparse.iarr",
            True,
            "test_expression_tsparse.iarr",
            marks=pytest.mark.heavy,
        ),
        # Use another order than before (precision needs to be relaxed a bit)
        (
            "ia.tan(t) * (ia.sin(x) + ia.cos(y)) + (ia.sqrt(z) * 2)",
            "np.tan(t) * (np.sin(x) + np.cos(y)) + (np.sqrt(z) * 2)",
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
    shape = [100, 200]
    chunks = [40, 50]
    bshape = [20, 20]

    xcfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=xcontiguous, urlpath=xurlpath)
    x32cfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=xcontiguous, urlpath="x32urlpath")
    ycfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=ycontiguous, urlpath=yurlpath)
    y32cfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=ycontiguous, urlpath="y32urlpath")
    zcfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=zcontiguous, urlpath=zurlpath)
    z32cfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=zcontiguous, urlpath="z32urlpath")
    tcfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=tcontiguous, urlpath=turlpath)
    t32cfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=tcontiguous, urlpath="t32urlpath")
    if xcfg.urlpath is None:
        x32cfg.urlpath = None
    if ycfg.urlpath is None:
        y32cfg.urlpath = None
    if zcfg.urlpath is None:
        z32cfg.urlpath = None
    if tcfg.urlpath is None:
        t32cfg.urlpath = None

    ia.remove_urlpath(xcfg.urlpath)
    ia.remove_urlpath(ycfg.urlpath)
    ia.remove_urlpath(x32cfg.urlpath)
    ia.remove_urlpath(y32cfg.urlpath)
    ia.remove_urlpath(zcfg.urlpath)
    ia.remove_urlpath(tcfg.urlpath)
    ia.remove_urlpath(z32cfg.urlpath)
    ia.remove_urlpath(t32cfg.urlpath)

    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        if dtype == np.float32:
            x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, cfg=x32cfg)
            y = ia.linspace(shape, 0.5, 1, dtype=dtype, cfg=y32cfg)
            z = ia.linspace(shape, 0.1, 0.9, dtype=dtype, cfg=z32cfg)
            t = ia.linspace(shape, 0.5, 1, dtype=dtype, cfg=t32cfg)
        else:
            x = ia.linspace(shape, 0.1, 0.9, dtype=dtype, cfg=xcfg)
            y = ia.linspace(shape, 0.5, 1, dtype=dtype, cfg=ycfg)
            z = ia.linspace(shape, 1.0, 2.0, dtype=dtype, cfg=zcfg)
            t = ia.linspace(shape, 1.5, 3.0, dtype=dtype, cfg=tcfg)

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

    ia.remove_urlpath(xcfg.urlpath)
    ia.remove_urlpath(ycfg.urlpath)
    ia.remove_urlpath(x32cfg.urlpath)
    ia.remove_urlpath(y32cfg.urlpath)
    ia.remove_urlpath(zcfg.urlpath)
    ia.remove_urlpath(tcfg.urlpath)
    ia.remove_urlpath(z32cfg.urlpath)
    ia.remove_urlpath(t32cfg.urlpath)


@pytest.mark.parametrize(
    "expression, contiguous, zcontiguous, zurlpath",
    [
        (
            "x + y",
            True,
            True,
            None,
        )
    ],
)
def test_chunks_blocks_params(expression, contiguous, zurlpath, zcontiguous):
    shape = [200]
    chunks = [40]
    blocks = [20]
    ia.remove_urlpath(zurlpath)

    # First with default chunks and blocks when operands chunks and blocks are equal
    xcfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=contiguous)
    ycfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=contiguous)
    zcfg = ia.Config(contiguous=zcontiguous, urlpath=zurlpath)

    x = ia.linspace(shape, 0.1, 0.2, cfg=xcfg)
    y = ia.linspace(shape, 0, 1, cfg=ycfg)

    expr = ia.expr_from_string(expression, {"x": x, "y": y}, cfg=zcfg)
    iout = expr.eval()
    assert iout.cfg.chunks == chunks
    assert iout.cfg.blocks == blocks
    ia.remove_urlpath(zcfg.urlpath)

    # Now with default chunks and blocks when operands chunks and blocks are not equal
    ycfg = ia.Config(chunks=[30], blocks=blocks, contiguous=contiguous)
    zcfg = ia.Config(contiguous=zcontiguous, urlpath=zurlpath)
    y = ia.linspace(shape, 0, 1, cfg=ycfg)
    expr = ia.expr_from_string(expression, {"x": x, "y": y}, cfg=zcfg)
    iout = expr.eval()
    assert iout.cfg.chunks != chunks
    assert iout.cfg.blocks != blocks
    ia.remove_urlpath(zcfg.urlpath)

    # Check that the provided chunks and blocks are used
    ycfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=contiguous)
    zcfg = ia.Config(chunks=[10], blocks=[5], contiguous=zcontiguous, urlpath=zurlpath)
    y = ia.linspace(shape, 0, 1, cfg=ycfg)
    expr = ia.expr_from_string(expression, {"x": x, "y": y}, cfg=zcfg)
    iout = expr.eval()
    assert iout.cfg.chunks == zcfg.chunks
    assert iout.cfg.blocks == zcfg.blocks


@pytest.mark.parametrize(
    "expression, operands",
    [
        ("x + y", ("x", "y")),
        ("(3 + a) -c -1", ("a", "c")),
        ("(3.+a)/c", ("a", "c")),
        (
            "2.3e9 + b + atan2(a + c / b) - 2**z + 12 % 3 * sin(9 - 9.9) - C",
            ("C", "a", "b", "c", "z"),
        ),
    ],
)
def test_get_operands(expression, operands):
    assert ia.expr_get_operands(expression) == operands


@pytest.mark.parametrize(
    "sexpr, sexpr_scalar, inputs",
    [
        ("x + 1", "x + y", {"x": ia.arange((10,)), "y": 1}),
        ("x + y + 1.35", "x + y + z", {"x": ia.arange((10,)), "y": 1, "z": 1.35}),
    ],
)
def test_scalar_params(sexpr, sexpr_scalar, inputs):
    expr = ia.expr_from_string(sexpr, inputs)
    expr_scalar = ia.expr_from_string(sexpr_scalar, inputs)
    out = expr.eval()
    out_udf = expr_scalar.eval()
    tol = 1e-14
    np.testing.assert_allclose(out.data, out_udf.data, rtol=tol, atol=tol)


# Expression evaluation from views
@pytest.mark.parametrize(
    "expr, np_expr, xcontiguous, xurlpath, ycontiguous, yurlpath, tcontiguous, turlpath, dtype, view_dtype",
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
            np.int16,
            np.float32,
        ),
        # transcendental functions
        (
            "ia.cos(x) + y",
            "np.cos(x) + y",
            True,
            "test_expression_xcontiguous.iarr",
            True,
            "test_expression_ycontiguous.iarr",
            True,
            None,
            np.bool_,
            np.int8,
        ),
        (
            "ia.sin(x) * ia.sin(x) + ia.cos(y)",
            "np.sin(x) * np.sin(x) + np.cos(y)",
            False,
            "test_expression_xsparse.iarr",
            True,
            "test_expression_xcontiguous.iarr",
            False,
            None,
            np.float32,
            np.float64,
        ),
    ],
)
def test_expr_type_view(
    expr,
    np_expr,
    xcontiguous,
    xurlpath,
    ycontiguous,
    yurlpath,
    tcontiguous,
    turlpath,
    dtype,
    view_dtype,
):
    if (view_dtype not in [np.float32, np.float64]
        and any(func in expr for func in ["cos", "sin", "tan", "acos", "asin", "atan"])
    ):
        pytest.skip("cannot compute this reduction with this dtype")

    shape = [200, 300]
    chunks = [40, 50]
    bshape = [20, 20]

    xcfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=xcontiguous, urlpath=xurlpath)
    ycfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=ycontiguous, urlpath=yurlpath)
    tcfg = ia.Config(chunks=chunks, blocks=bshape, contiguous=tcontiguous, urlpath=turlpath)

    ia.remove_urlpath(xcfg.urlpath)
    ia.remove_urlpath(ycfg.urlpath)
    ia.remove_urlpath(tcfg.urlpath)

    # The ranges below are important for not overflowing operations
    x_ = ia.linspace(shape, 0.1, 0.9, dtype=dtype, cfg=xcfg)
    y_ = ia.linspace(shape, 0.5, 1, dtype=dtype, cfg=ycfg)
    t_ = ia.linspace(shape, 0.5, 1, dtype=dtype, cfg=tcfg)
    x = x_.astype(view_dtype)
    y = y_.astype(view_dtype)
    t = t_.astype(view_dtype)

    # NumPy computation
    npx = ia.iarray2numpy(x)
    npy = ia.iarray2numpy(y)
    npt = ia.iarray2numpy(t)
    npout = eval("%s" % np_expr, {"np": np, "x": npx, "y": npy, "t": npt})

    # High-level ironarray eval
    lazy_expr = eval(expr, {"ia": ia, "x": x, "y": y, "t": t})
    iout2 = lazy_expr.eval()
    npout2 = ia.iarray2numpy(iout2)

    tol = 1e-6 if dtype is np.float32 else 1e-14
    if dtype in [np.float32, np.float64]:
        np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)
    else:
        np.testing.assert_array_equal(npout, npout2)

    ia.remove_urlpath(xcfg.urlpath)
    ia.remove_urlpath(ycfg.urlpath)
    ia.remove_urlpath(tcfg.urlpath)
