import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, contiguous, urlpath, mode",
    [
        ([100, 100], [50, 50], [20, 20], np.float32, False, None, "r"),
        ([100, 100], [20, 20], [10, 10], np.float64, True, "test_transpose_contiguous.iarr", "r+"),
        ([100, 500], [50, 70], [20, 20], np.float32, False, "test_transpose_sparse.iarr", "w"),
        ([50, 26], [20, 10], [15, 5], np.float64, True, None, "w-"),
        pytest.param([1453, 266], [100, 200], [30, 20], np.float64, True, None, "w-", marks=pytest.mark.heavy),
    ],
)
def test_transpose(shape, chunks, blocks, dtype, contiguous, urlpath, mode):
    ia.remove_urlpath(urlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=contiguous, urlpath=urlpath)

    a = ia.linspace(shape, -10, 10, cfg=cfg, dtype=dtype)

    b = ia.iarray2numpy(a)
    bn = b.T

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    at = a.T
    an = ia.iarray2numpy(at)
    np.testing.assert_allclose(an, bn, rtol=rtol)

    if mode in ["r", "r+"]:
        with pytest.raises(IOError):
            at = a.transpose(mode=mode)
        at = a.transpose()
    else:
        at = a.transpose(mode=mode)
    an = ia.iarray2numpy(at)
    np.testing.assert_allclose(an, bn, rtol=rtol)

    if mode in ["r", "r+"]:
        with pytest.raises(IOError):
            at = ia.transpose(a, mode=mode)
        at = ia.transpose(a)
    else:
        at = ia.transpose(a, mode=mode)

    an = ia.iarray2numpy(at)
    np.testing.assert_allclose(an, bn, rtol=rtol)

    ia.remove_urlpath(urlpath)
