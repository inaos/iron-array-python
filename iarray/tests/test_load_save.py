import pytest
import iarray as ia
import numpy as np


# Test load and save
@pytest.mark.parametrize("shape, pshape, dtype, load_in_mem",
                         [
                             ([100], [20], np.float64, True),
                             ([100, 100], [20, 20], np.float32, True),
                             ([100, 100], [5, 10], np.float64, False),
                             ([100, 100, 100], [5, 46, 10], np.float32, False)

                         ])
def test_load_save(shape, pshape, dtype, load_in_mem):

    a = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10)
    an = ia.iarray2numpy(a)

    ia.save(a, "test_load_save.iarray")

    b = ia.load("test_load_save.iarray", load_in_mem=load_in_mem)
    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(an, bn)
