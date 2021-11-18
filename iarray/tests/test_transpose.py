import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        ([100, 100], [50, 50], [20, 20], np.float32, False, None),
        ([100, 100], [20, 20], [10, 10], np.float64, True, "test_transpose_contiguous.iarr"),
        ([100, 500], [50, 70], [20, 20], np.float32, False, "test_transpose_sparse.iarr"),
        ([1453, 266], [100, 200], [30, 20], np.float64, True, None),
    ],
)
def test_transpose(shape, chunks, blocks, dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    store = ia.Store(chunks, blocks, contiguous=contiguous, urlpath=urlpath)

    a = ia.linspace(shape, -10, 10, store=store, dtype=dtype)

    b = ia.iarray2numpy(a)
    bn = b.T

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    at = a.T
    an = ia.iarray2numpy(at)
    np.testing.assert_allclose(an, bn, rtol=rtol)

    at = a.transpose()
    an = ia.iarray2numpy(at)
    np.testing.assert_allclose(an, bn, rtol=rtol)

    at = ia.transpose(a)

    an = ia.iarray2numpy(at)
    np.testing.assert_allclose(an, bn, rtol=rtol)

    ia.remove_urlpath(urlpath)
