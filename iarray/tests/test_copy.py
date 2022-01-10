import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "shape, chunks, blocks",
    [
        ([100, 100], [50, 50], [20, 20]),
        ([20, 100, 30, 50], [10, 40, 10, 11], [4, 5, 3, 7]),
        ([10, 13, 12, 14, 12, 10], [5, 4, 6, 2, 3, 7], [2, 2, 2, 2, 2, 2]),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.int32, np.uint64, np.uint32])
@pytest.mark.parametrize(
    " contiguous, urlpath, urlpath2",
    [
        (False, None, None),
        (False, None, None),
        (True, None, None),
        (True, "test_copy.iarr", "test_copy2.iarr"),
        (False, "test_copy.iarr", "test_copy2.iarr"),
    ],
)
def test_copy(shape, chunks, blocks, dtype, contiguous, urlpath, urlpath2):
    ia.remove_urlpath(urlpath)
    ia.remove_urlpath(urlpath2)
    max = 1
    if dtype not in [np.float64, np.float32]:
        for i in range(len(shape)):
            max *= shape[i]
    a_ = ia.arange(shape, 0, max, dtype=dtype, chunks=chunks, blocks=blocks, contiguous=contiguous, urlpath=urlpath)
    sl = tuple([slice(0, s - 1) for s in shape])
    a = a_[sl]
    b = a.copy(urlpath=urlpath2)
    an = ia.iarray2numpy(a)
    bn = ia.iarray2numpy(b)

    if dtype in [np.float64, np.float32]:
        rtol = 1e-6 if dtype == np.float32 else 1e-14
        np.testing.assert_allclose(an, bn, rtol=rtol)
    else:
        np.testing.assert_array_equal(an, bn)

    c = a.copy(favor=ia.Favor.SPEED)
    assert c.cfg.btune is True
    with pytest.raises(ValueError):
        a.copy(favor=ia.Favor.CRATIO, btune=False)
    d = a.copy()
    assert d.cfg.btune is False

    with pytest.raises(IOError):
        a.copy(mode="r")
    with pytest.raises(IOError):
        a.copy(mode="r+")

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath(urlpath2)
