import pytest
import iarray as ia
import numpy as np
import os


# Test load, open and save
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
def test_load_save(shape, chunks, blocks, dtype, func):
    urlpath = "test_load_save.iarray"

    if os.path.exists(urlpath):
        os.remove(urlpath)

    store = ia.Store(chunks, blocks)
    a = ia.linspace(shape, -10, 10, dtype=dtype, store=store)
    an = ia.iarray2numpy(a)

    ia.save(urlpath, a)

    b = func(urlpath)
    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(an, bn)

    if os.path.exists(urlpath):
        os.remove(urlpath)
