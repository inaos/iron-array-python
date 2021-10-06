import pytest
import iarray as ia
import numpy as np


# Test load, open and save
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    [
        ([123], [44], [20]),
        ([100, 123], [12, 21], [10, 10]),
        ([100, 100], [5, 17], [5, 5]),
        ([104, 121, 212], [5, 46, 10], [2, 8, 5]),
    ],
)
@pytest.mark.parametrize("func", [ia.load, ia.open])
def test_load_save(shape, chunks, blocks, dtype, func, contiguous):
    urlpath = "test_load_save.iarr"

    ia.remove_urlpath(urlpath)

    store = ia.Store(chunks, blocks, contiguous=contiguous)
    a = ia.linspace(shape, -10, 10, dtype=dtype, store=store)
    an = ia.iarray2numpy(a)

    ia.save(urlpath, a, contiguous=contiguous)

    b = func(urlpath)
    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(an, bn)

    ia.remove_urlpath(urlpath)
