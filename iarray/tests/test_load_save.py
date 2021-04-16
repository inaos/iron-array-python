import pytest
import iarray as ia
import numpy as np


# Test load, open and save
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape, chunkshape, blockshape",
    [
        ([123], [44], [20]),
        ([100, 123], [12, 21], [10, 10]),
        ([100, 100], [5, 17], [5, 5]),
        ([104, 121, 212], [5, 46, 10], [2, 8, 5]),
    ],
)
@pytest.mark.parametrize("func", [ia.load, ia.open])
def test_load_save(shape, chunkshape, blockshape, dtype, func):

    storage = ia.Storage(chunkshape, blockshape)
    a = ia.linspace(shape, -10, 10, dtype=dtype, storage=storage)
    an = ia.iarray2numpy(a)

    ia.save("test_load_save.iarray", a)

    b = func("test_load_save.iarray")
    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(an, bn)
