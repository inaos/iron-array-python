import pytest
import iarray as ia
import numpy as np
from math import isclose

params_names = "shape, chunks, blocks, axis, dtype"
params_data = [
    ([100, 100], [50, 50], [20, 20], 0, np.float64),
    ([10, 10, 10], [5, 5, 4], [5, 5, 2], 1, np.float32),
    pytest.param(
        [20, 100, 30, 50],
        [10, 40, 10, 11],
        [4, 5, 3, 7],
        1,
        np.float64,
        marks=pytest.mark.heavy,
    ),
    ([40, 45], [20, 23], [9, 7], 0, np.int64),
    pytest.param(
        [70, 45, 56, 34],
        [20, 23, 30, 34],
        [9, 7, 8, 7],
        (0, 3),
        np.int64,
        marks=pytest.mark.heavy,
    ),
    ([30], [20], [20], None, np.uint32),
    ([10, 10, 10], [4, 4, 4], [2, 2, 2], 1, np.bool_),
]


@pytest.mark.parametrize(params_names, params_data)
@pytest.mark.parametrize("rfunc", ["mean", "sum", "prod", "max", "min", "median", "var", "std"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "test_reduce.iarr"])
@pytest.mark.parametrize("view", [True, False])
@pytest.mark.parametrize("nan", [True, False])
def test_reduce(shape, chunks, blocks, axis, dtype, rfunc, contiguous, urlpath, view, nan):

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")

    if dtype not in [np.float32, np.float64] and nan is True:
        pytest.skip("cannot compute this reduction with this dtype")

    a1 = ia.linspace(
        shape, 0, 100, chunks=chunks, blocks=blocks, urlpath=urlpath, mode="w", dtype=dtype
    )

    if nan:
        slices = tuple([slice(np.random.randint(1, s)) for s in a1.shape])
        a1[slices] = np.nan

    if view:
        slices = tuple([slice(np.random.randint(1, s)) for s in a1.shape])
        a1 = a1[slices]

    a2 = a1.data

    b2 = getattr(np, rfunc)(a2, axis=axis)
    if urlpath:
        b1 = getattr(ia, rfunc)(a1, axis=axis, urlpath="test_reduce_res.iarr")
    else:
        b1 = getattr(ia, rfunc)(a1, axis=axis)

    if b2.ndim == 0:
        isclose(b1, b2)
    else:
        tol = 1e-5 if dtype is np.float32 else 1e-14
        np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=tol, atol=tol)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")


params_names = "shape, chunks, blocks, axis, dtype"
params_data = [
    ([30], [8], [4], 0, np.float32),
    ([40, 45], [20, 23], [9, 7], 0, np.float32),
    pytest.param(
        [70, 45, 56, 34],
        [20, 23, 30, 34],
        [9, 7, 8, 7],
        (0, 3),
        np.float64,
        marks=pytest.mark.heavy,
    ),
    ([10, 10, 10], [5, 5, 4], [5, 5, 2], 1, np.float64),
]


@pytest.mark.parametrize(params_names, params_data)
@pytest.mark.parametrize(
    "rfunc", ["nanmean", "nansum", "nanprod", "nanmax", "nanmin", "nanmedian", "nanvar", "nanstd"]
)
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "test_reduce.iarr"])
@pytest.mark.parametrize("view", [True, False])
@pytest.mark.parametrize("nan", [True, False])
def test_reduce_nan(shape, chunks, blocks, axis, dtype, rfunc, contiguous, urlpath, view, nan):

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_nan_res.iarr")

    a1 = ia.linspace(
        shape, 0, 100, chunks=chunks, blocks=blocks, urlpath=urlpath, mode="w", dtype=dtype
    )

    if nan:
        slices = tuple([slice(np.random.randint(1, s)) for s in a1.shape])
        a1[slices] = np.nan

    if view:
        slices = tuple([slice(np.random.randint(1, s)) for s in a1.shape])
        a1 = a1[slices]

    a2 = a1.data

    b2 = getattr(np, rfunc)(a2, axis=axis)
    if urlpath:
        b1 = getattr(ia, rfunc)(a1, axis=axis, urlpath="test_reduce_nan_res.iarr")
    else:
        b1 = getattr(ia, rfunc)(a1, axis=axis)

    if b2.ndim == 0:
        isclose(b1, b2)
    else:
        tol = 1e-5 if dtype is np.float32 else 1e-14
        np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=tol, atol=tol)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_nan_res.iarr")


params_names = "shape, chunks, blocks, axis, dtype, view_dtype"
params_data = [
    ([100, 100], [50, 50], [20, 20], 0, np.float64, np.uint64),
    pytest.param(
        [45, 56, 34],
        [23, 30, 34],
        [7, 8, 7],
        (0, 2),
        np.int64,
        np.float64,
        marks=pytest.mark.heavy,
    ),
    ([10, 10, 10], [4, 4, 4], [2, 2, 2], 1, np.bool_, np.uint16),
]


@pytest.mark.parametrize(params_names, params_data)
@pytest.mark.parametrize("rfunc", ["mean", "sum", "prod", "max", "min"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "test_reduce.iarr"])
def test_red_type_view(shape, chunks, blocks, axis, dtype, view_dtype, rfunc, contiguous, urlpath):

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")

    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=contiguous, urlpath=urlpath)
    a1 = ia.linspace(shape, 0, 100, dtype=dtype, cfg=cfg)
    a2 = a1.astype(view_dtype)

    b2 = getattr(np, rfunc)(a2.data, axis=axis)
    b1 = getattr(ia, rfunc)(a2, axis=axis, urlpath="test_reduce_res.iarr")

    if view_dtype in [np.float64, np.float32] or rfunc == "mean":
        rtol = 1e-6 if view_dtype == np.float32 else 1e-14
        if b2.ndim == 0:
            isclose(b1, b2, rel_tol=rtol, abs_tol=0.0)
        else:
            np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol, atol=0)
    else:
        if b2.ndim == 0:
            assert b1 == b2
        else:
            np.testing.assert_array_equal(ia.iarray2numpy(b1), b2)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")
