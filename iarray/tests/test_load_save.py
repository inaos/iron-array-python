import pytest
import iarray as ia
import numpy as np


# Test load and save
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([100], [20], np.float64),
                             ([100, 100], [20, 20], np.float32),
                             ([100, 100], [5, 10], np.float64),
                             ([100, 100, 100], [5, 46, 10], np.float32)

                         ])
def test_load_save(shape, pshape, dtype):

    a = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10)
    an = ia.iarray2numpy(a)

    ia.save(a, "test_load_save.iarray")

    b = ia.load("test_load_save.iarray")
    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(an, bn)
