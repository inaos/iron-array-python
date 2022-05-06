import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath, mode",
    [
        ([100, 100], [50, 50], [20, 20], np.float32, None, False, None, "r"),
        ([100, 100], [20, 20], [10, 10], np.float64, '>f4', True, "test_transpose_contiguous.iarr", "r+"),
        ([100, 500], [50, 70], [20, 20], np.float32, '>f2', False, "test_transpose_sparse.iarr", "w"),
        ([50, 26], [20, 10], [15, 5], np.float64, 'i4', True, None, "w-"),
        pytest.param([1453, 266], [100, 200], [30, 20], np.float64, None, True, None, "w-", marks=pytest.mark.heavy),
    ],
)
def test_transpose(shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath, mode):
    ia.remove_urlpath(urlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=contiguous, urlpath=urlpath)

    a = ia.linspace(shape, -10, 10, cfg=cfg, dtype=dtype, np_dtype=np_dtype)

    b = ia.iarray2numpy(a)
    bn = b.T

    npdtype = dtype if np_dtype is None else np.dtype(np_dtype)
    rtol = 1e-6 if npdtype == np.float32 else 1e-14

    at = a.T
    an = ia.iarray2numpy(at)
    if npdtype in [np.float16, np.float32, np.float64]:
        np.testing.assert_allclose(an, bn, rtol=rtol)
    else:
        np.testing.assert_equal(an, bn)

    if mode in ["r", "r+"]:
        with pytest.raises(IOError):
            at = a.transpose(mode=mode)
        at = a.transpose()
    else:
        at = a.transpose(mode=mode)
    an = ia.iarray2numpy(at)
    if npdtype in [np.float16, np.float32, np.float64]:
        np.testing.assert_allclose(an, bn, rtol=rtol)
    else:
        np.testing.assert_equal(an, bn)

    if mode in ["r", "r+"]:
        with pytest.raises(IOError):
            at = ia.transpose(a, mode=mode)
        at = ia.transpose(a)
    else:
        at = ia.transpose(a, mode=mode)

    an = ia.iarray2numpy(at)
    if npdtype in [np.float16, np.float32, np.float64]:
        np.testing.assert_allclose(an, bn, rtol=rtol)
    else:
        np.testing.assert_equal(an, bn)

    ia.remove_urlpath(urlpath)
