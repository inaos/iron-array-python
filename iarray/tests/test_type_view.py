import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "dtype, view_dtype",
    [
        (np.dtype(np.float32), np.dtype(np.int64)),
        (np.dtype(np.uint64), np.float64),
        (np.int64, np.dtype(np.float64)),
        (np.int8, np.bool_),
        (np.bool_, np.float32),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, contiguous, urlpath",
    [
        pytest.param(
            [123, 432, 222], [24, 31, 15], [24, 31, 15], True, None, marks=pytest.mark.heavy
        ),
        ([567, 375], [52, 16], [52, 16], False, "test_type_sparse.iarr"),
        ([10, 12, 5], [5, 5, 5], [5, 5, 5], True, "test_type_contiguous.iarr"),
        ([12, 16], [12, 16], [3, 5], False, None),
    ],
)
def test_type(shape, chunks, blocks, dtype, view_dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)

    size = np.prod(shape)
    if dtype == np.bool_:
        a = ia.full(shape, True, dtype=dtype)
    else:
        a = ia.arange(0, size, shape=shape, dtype=dtype)
    b = ia.iarray2numpy(a)
    assert not a.is_view
    c = ia.astype(a, view_dtype)
    assert c.is_view
    c = ia.iarray2numpy(c)
    d = b.astype(view_dtype)

    if view_dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(c, d)
    else:
        np.testing.assert_array_equal(c, d)

    ia.remove_urlpath(urlpath)


# Slice + type views
slice_data = [
    ([30, 100], [20, 20], [10, 13], True, None),
    ([30, 130], [50, 50], [20, 25], False, None),
    pytest.param(
        [10, 78, 55, 21],
        [3, 30, 30, 21],
        [3, 12, 6, 21],
        True,
        "test_slice_acontiguous.iarr",
        marks=pytest.mark.heavy,
    ),
    ([30, 100], [30, 44], [30, 2], False, "test_slice_asparse.iarr"),
]


@pytest.mark.parametrize(
    "dtype, view_dtype",
    [
        (np.float32, np.uint64),
        (np.uint64, np.float64),
        (np.uint8, np.bool_),
        (np.int16, np.uint32),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, acontiguous, aurlpath",
    slice_data,
)
def test_slice_type(shape, chunks, blocks, acontiguous, aurlpath, dtype, view_dtype):
    ia.remove_urlpath(aurlpath)
    cfg = ia.Config(
        chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath, nthreads=1
    )
    max = 1
    if dtype not in [np.float64, np.float32]:
        for i in range(len(shape)):
            max *= shape[i]
    a = ia.arange(0, max, shape=shape, cfg=cfg, mode="w", dtype=dtype)
    an = ia.iarray2numpy(a)

    slices = tuple([slice(0, s - 1) for s in shape])
    a[slices] = 0
    an[slices] = 0
    if dtype in [np.float32, np.float64]:
        np.testing.assert_almost_equal(a.data, an)
    else:
        np.testing.assert_equal(a.data, an)

    b_ = a[slices]
    c = ia.astype(b_, view_dtype)

    d_ = an[slices]
    d = d_.astype(view_dtype)

    bn = ia.iarray2numpy(c)
    assert d.shape == bn.shape
    assert d.ndim == bn.ndim
    if view_dtype in [np.float32, np.float64]:
        np.testing.assert_almost_equal(d, bn)
    else:
        np.testing.assert_equal(d, bn)

    ia.remove_urlpath(aurlpath)
