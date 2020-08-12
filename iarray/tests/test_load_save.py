import pytest
import iarray as ia
import numpy as np


# Test load and save
@pytest.mark.parametrize("shape, chunkshape, blockshape, dtype, load_in_mem",
                         [
                             ([123], [44], [20], np.float64, True),
                             ([100, 123], [12, 21], [10, 10], np.float32, True),
                             ([100, 100], [5, 17], [5, 5], np.float64, False),
                             ([104, 121, 212], [5, 46, 10], [2, 8, 5], np.float32, False)

                         ])
def test_load_save(shape, chunkshape, blockshape, dtype, load_in_mem):

    storage = ia.StorageProperties("blosc", chunkshape, blockshape)
    a = ia.linspace(ia.dtshape(shape, dtype), -10, 10, storage=storage)
    an = ia.iarray2numpy(a)

    ia.save(a, "test_load_save.iarray")

    b = ia.load("test_load_save.iarray", load_in_mem=load_in_mem)
    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(an, bn)
