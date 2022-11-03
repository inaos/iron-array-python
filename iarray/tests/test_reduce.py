import pytest
import iarray as ia
import numpy as np

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
@pytest.mark.parametrize(
    "rfunc, correction",
    [
        ("mean", None),
        ("sum", None),
        ("prod", None),
        ("max", None),
        ("min", None),
        ("median", None),
        ("var", 0),
        ("var", 1),
        ("var", 10),
        ("std", 0),
        ("std", 1),
        ("std", 23),
    ],
)
@pytest.mark.parametrize("contiguous", [True])
@pytest.mark.parametrize("urlpath", [None])
@pytest.mark.parametrize("view", [True, False])
@pytest.mark.parametrize("nan", [True, False])
def test_reduce(
    shape, chunks, blocks, axis, dtype, rfunc, correction, contiguous, urlpath, view, nan
):

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")

    if (
        (
            dtype not in [np.float32, np.float64]
            and (nan is True or rfunc in ["mean", "var", "std"])
        )
        or dtype == ia.bool
        and rfunc in ["sum", "prod", "max", "min"]
    ):
        pytest.skip("cannot compute this reduction with this dtype")

    a1 = ia.linspace(
        0,
        100,
        int(np.prod(shape)),
        shape=shape,
        chunks=chunks,
        blocks=blocks,
        urlpath=urlpath,
        mode="w",
        dtype=dtype,
    )

    if nan:
        slices = tuple([slice(np.random.randint(1, s)) for s in a1.shape])
        a1[slices] = np.nan

    if view:
        slices = tuple([slice(np.random.randint(1, s)) for s in a1.shape])
        a1 = a1[slices]

    a2 = a1.data

    if rfunc in ["var", "std"]:
        b2 = getattr(np, rfunc)(a2, axis=axis, ddof=correction)
        if urlpath:
            b1 = getattr(ia, rfunc)(
                a1, axis=axis, correction=correction, urlpath="test_reduce_res.iarr"
            )
        else:
            b1 = getattr(ia, rfunc)(a1, axis=axis, correction=correction)
    else:
        b2 = getattr(np, rfunc)(a2, axis=axis)
        if urlpath:
            b1 = getattr(ia, rfunc)(a1, axis=axis, urlpath="test_reduce_res.iarr")
        else:
            b1 = getattr(ia, rfunc)(a1, axis=axis)

    tol = 1e-5 if dtype is np.float32 else 1e-14
    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=tol, atol=tol)

    if rfunc in ["mean", "sum", "prod", "max", "min"]:
        if urlpath:
            b1 = getattr(ia, rfunc)(
                a1, axis=axis, oneshot=True, mode="w", urlpath="test_reduce_res.iarr"
            )
        else:
            b1 = getattr(ia, rfunc)(a1, axis=axis, oneshot=True)

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
    "rfunc, correction",
    [
        ("nanmean", None),
        ("nansum", None),
        ("nanprod", None),
        ("nanmax", None),
        ("nanmin", None),
        ("nanmedian", None),
        ("nanvar", 0.0),
        ("nanvar", 1),
        ("nanstd", 0.0),
        ("nanstd", 1.0),
    ],
)
@pytest.mark.parametrize("contiguous", [False])
@pytest.mark.parametrize("urlpath", [None])
@pytest.mark.parametrize("view", [True, False])
@pytest.mark.parametrize("nan", [True, False])
def test_reduce_nan(
    shape, chunks, blocks, axis, dtype, rfunc, correction, contiguous, urlpath, view, nan
):

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_nan_res.iarr")

    a1 = ia.linspace(
        0,
        100,
        int(np.prod(shape)),
        shape=shape,
        chunks=chunks,
        blocks=blocks,
        urlpath=urlpath,
        mode="w",
        dtype=dtype,
    )

    if nan:
        slices = tuple([slice(np.random.randint(1, s)) for s in a1.shape])
        a1[slices] = np.nan

    if view:
        slices = tuple([slice(np.random.randint(1, s)) for s in a1.shape])
        a1 = a1[slices]

    a2 = a1.data

    if rfunc in ["nanvar", "nanstd"]:
        b2 = getattr(np, rfunc)(a2, axis=axis, ddof=correction)
        if urlpath:
            b1 = getattr(ia, rfunc)(
                a1, axis=axis, correction=correction, urlpath="test_reduce_nan_res.iarr"
            )
        else:
            b1 = getattr(ia, rfunc)(a1, axis=axis, correction=correction)
    else:
        b2 = getattr(np, rfunc)(a2, axis=axis)
        if urlpath:
            b1 = getattr(ia, rfunc)(a1, axis=axis, urlpath="test_reduce_nan_res.iarr")
        else:
            b1 = getattr(ia, rfunc)(a1, axis=axis)

    tol = 1e-5 if dtype is np.float32 else 1e-14
    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=tol, atol=tol)

    if rfunc in ["nansum", "nanprod", "nanmax", "nanmin"]:
        if urlpath:
            b1 = getattr(ia, rfunc)(
                a1, axis=axis, oneshot=True, mode="w", urlpath="test_reduce_nan_res.iarr"
            )
        else:
            b1 = getattr(ia, rfunc)(
                a1,
                axis=axis,
                oneshot=True,
            )

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

    if (
        (view_dtype not in [np.float32, np.float64] and rfunc in ["mean", "var", "std"])
        or view_dtype == ia.bool
        and rfunc in ["sum", "prod"]
    ):
        pytest.skip("cannot compute this reduction with this dtype")
    elif view_dtype == ia.bool and rfunc in ["max", "min"]:
        rfunc = "any" if rfunc == "max" else "all"
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=contiguous, urlpath=urlpath)
    a1 = ia.linspace(0, 100, int(np.prod(shape)), shape=shape, dtype=dtype, cfg=cfg)
    a2 = ia.astype(a1, view_dtype)

    b2 = getattr(np, rfunc)(a2.data, axis=axis)
    b1 = getattr(ia, rfunc)(a2, axis=axis, urlpath="test_reduce_res.iarr")

    if view_dtype in [np.float64, np.float32] or rfunc == "mean":
        rtol = 1e-6 if view_dtype == np.float32 else 1e-14
        np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol, atol=0)
    else:
        np.testing.assert_array_equal(ia.iarray2numpy(b1), b2)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")


params_names = "shape, chunks, blocks, axis, dtype"
params_data = [
    ([100, 100], [50, 50], [20, 20], 0, np.float64),
    ([10, 10, 10], [4, 4, 4], [2, 2, 2], 1, np.float32),
    ([30], [20], [20], None, np.float32),
]


@pytest.mark.parametrize(params_names, params_data)
@pytest.mark.parametrize(
    "rfunc",
    [
        "mean",
        "sum",
        "prod",
        "max",
        "min",
        "median",
        "var",
        "std",
        "nanmean",
        "nansum",
        "nanprod",
        "nanmax",
        "nanmin",
        "nanmedian",
        "nanvar",
        "nanstd",
    ],
)
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "test_reduce.iarr"])
def test_reduce_storage(shape, chunks, blocks, axis, dtype, rfunc, contiguous, urlpath):

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")

    a1 = ia.linspace(
        0,
        100,
        int(np.prod(shape)),
        shape=shape,
        chunks=chunks,
        blocks=blocks,
        urlpath=urlpath,
        mode="w",
        dtype=dtype,
    )
    a2 = a1.data

    b2 = getattr(np, rfunc)(a2, axis=axis)
    if urlpath:
        b1 = getattr(ia, rfunc)(a1, axis=axis, urlpath="test_reduce_res.iarr")
    else:
        b1 = getattr(ia, rfunc)(a1, axis=axis)

    tol = 1e-5 if dtype is np.float32 else 1e-14
    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=tol, atol=tol)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")
