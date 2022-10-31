import pytest
import numpy as np
import iarray as ia


@pytest.mark.parametrize(
    "func, np_func, nan, nargs",
    [
        (ia.all, np.all, False, 1),
        (ia.any, np.any, False, 1),
        (ia.max, np.max, False, 1),
        (ia.min, np.min, False, 1),
        (ia.sum, np.sum, False, 1),
        (ia.prod, np.prod, False, 1),
        (ia.mean, np.mean, False, 1),
        (ia.std, np.std, False, 1),
        (ia.var, np.var, False, 1),
        (ia.median, np.median, False, 1),
        (ia.nanmax, np.nanmax, True, 1),
        (ia.nanmin, np.nanmin, True, 1),
        (ia.nansum, np.nansum, True, 1),
        (ia.nanprod, np.nanprod, True, 1),
        (ia.nanmean, np.nanmean, True, 1),
        (ia.nanstd, np.nanstd, True, 1),
        (ia.nanvar, np.nanvar, True, 1),
        (ia.nanmedian, np.nanmedian, True, 1),
        (ia.abs, np.abs, False, 1),
        (ia.add, np.add, False, 2),
        (ia.acos, np.arccos, False, 1),
        (ia.asin, np.arcsin, False, 1),
        (ia.atan, np.arctan, False, 1),
        (ia.atan2, np.arctan2, False, 2),
        (ia.ceil, np.ceil, False, 1),
        (ia.cos, np.cos, False, 1),
        (ia.cosh, np.cosh, False, 1),
        (ia.divide, np.divide, False, 2),
        (ia.equal, np.equal, False, 2),
        (ia.exp, np.exp, False, 1),
        (ia.expm1, np.expm1, False, 1),
        (ia.floor, np.floor, False, 1),
        (ia.greater, np.greater, False, 2),
        (ia.greater_equal, np.greater_equal, False, 2),
        (ia.less, np.less, False, 2),
        (ia.less_equal, np.less_equal, False, 2),
        (ia.log, np.log, False, 1),
        (ia.log1p, np.log1p, False, 1),
        (ia.log10, np.log10, False, 1),
        (ia.logaddexp, np.logaddexp, False, 2),
        (ia.multiply, np.multiply, False, 2),
        (ia.negative, np.negative, False, 1),
        (ia.not_equal, np.not_equal, False, 2),
        (ia.positive, np.positive, False, 1),
        (ia.pow, np.power, False, 2),
        (ia.sqrt, np.sqrt, False, 1),
        (ia.square, np.square, False, 1),
        (ia.subtract, np.subtract, False, 2),
        (ia.sin, np.sin, False, 1),
        (ia.sinh, np.sinh, False, 1),
        (ia.tan, np.tan, False, 1),
        (ia.tanh, np.tanh, False, 1),
    ],
)
@pytest.mark.parametrize(
    "start, stop, shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        pytest.param(
            0.001,
            1,
            [20, 90, 50],
            [10, 21, 34],
            [5, 13, 7],
            ia.float64,
            False,
            "test_math_sparse.iarr",
            marks=pytest.mark.heavy,
        ),
        (0.2, 0.1, [4, 3, 5, 2], [4, 3, 5, 2], [2, 3, 2, 2], ia.float32, True, None),
    ],
)
def test_math(
    func, np_func, nan, nargs, start, stop, shape, chunks, blocks, dtype, contiguous, urlpath
):
    size = int(np.prod(shape))
    ia.remove_urlpath(urlpath)
    if func in [ia.all, ia.any]:
        a = ia.full(
            fill_value=True,
            shape=shape,
            dtype=ia.bool,
            chunks=chunks,
            blocks=blocks,
            contiguous=contiguous,
            urlpath=urlpath,
        )
        c = ia.iarray2numpy(a)
    else:
        a = ia.linspace(
            start,
            stop,
            int(np.prod(shape)),
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            blocks=blocks,
            contiguous=contiguous,
            urlpath=urlpath,
        )
        c = np.linspace(start, stop, size, dtype=dtype).reshape(shape)

    if nargs == 2:
        b = ia.linspace(
            start + 1,
            stop + 1,
            int(np.prod(shape)),
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            blocks=blocks,
        )
        d = np.linspace(start + 1, stop + 1, size, dtype=dtype).reshape(shape)
        ia_res = func(a, b)
        if not isinstance(ia_res, ia.IArray):
            ia_res = ia_res.eval()
        np_res = np_func(c, d)
    else:
        if nan:
            slices = tuple([slice(np.random.randint(1, s)) for s in a.shape])
            a[slices] = ia.nan
            c[slices] = np.nan
        ia_res = func(a)
        if not isinstance(ia_res, ia.IArray):
            ia_res = ia_res.eval()
        np_res = np_func(c)
    if np_res.dtype == np.bool_:
        ia_res.np_dtype = np_res.dtype
    if np_res.dtype in [np.float32, np.float64]:
        tol = 1e-6 if dtype is np.float32 else 1e-14
        np.testing.assert_allclose(ia_res.data, np_res, rtol=tol, atol=tol)
    else:
        np.testing.assert_equal(ia_res.data, np_res)

    ia.remove_urlpath(urlpath)
