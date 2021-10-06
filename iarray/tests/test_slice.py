import pytest
import iarray as ia
import numpy as np
from math import isclose


# Slice


@pytest.mark.parametrize(
    "slices",
    [
        slice(10, 20),
        60,
        (slice(5, 30), slice(10, 40)),
        (slice(5, 5), slice(5, 23)),
        (slice(5, 5), slice(33, 12)),
        (slice(5, 30), 47, ...),
        (..., slice(5, 6)),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape, chunks, blocks, acontiguous, aurlpath",
    [
        ([100, 100], [20, 20], [10, 13], True, None),
        ([100, 130], None, None, False, None),
        ([98, 78, 55, 21], None, None, True, "test_slice_acontiguous.iarr"),
        ([100, 100], [30, 44], [30, 2], False, "test_slice_asparse.iarr"),
    ],
)
def test_slice(slices, shape, chunks, blocks, dtype, acontiguous, aurlpath):
    ia.remove(aurlpath)
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks, contiguous=acontiguous, urlpath=aurlpath)

    a = ia.linspace(shape, -10, 10, store=store, dtype=dtype)
    an = ia.iarray2numpy(a)

    a[slices] = 0
    an[slices] = 0
    np.testing.assert_almost_equal(a.data, an)

    data = ia.arange(shape, dtype=dtype)[slices]

    a[slices] = data
    an[slices] = data.data
    np.testing.assert_almost_equal(a.data, an)

    b = a[slices]
    an2 = an[slices]

    if b.ndim == 0:
        isclose(an2, b)
    else:
        bn = ia.iarray2numpy(b)
        assert an2.shape == bn.shape
        assert an2.ndim == bn.ndim
        np.testing.assert_almost_equal(an[slices], bn)

    ia.remove(aurlpath)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape, chunks, blocks, acontiguous, aurlpath",
    [
        ([100, 100], [20, 20], [10, 13], True, None),
        ([100, 130], None, None, False, None),
        ([98, 78, 55, 21], None, None, True, "test_slice_acontiguous.iarr"),
        ([100, 100], [30, 44], [30, 2], False, "test_slice_asparse.iarr"),
    ],
)
def test_double_slice(shape, chunks, blocks, dtype, acontiguous, aurlpath):
    ia.remove(aurlpath)
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks, contiguous=acontiguous, urlpath=aurlpath)

    a = ia.linspace(shape, -10, 10, store=store, dtype=dtype)
    b1 = a[4]
    b2 = a[4]
    np.testing.assert_almost_equal(b1.data, b2.data)

    ia.remove(aurlpath)
