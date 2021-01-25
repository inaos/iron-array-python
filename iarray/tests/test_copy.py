import pytest
import iarray as ia
import numpy as np
import os


@pytest.mark.parametrize(
    "shape, chunkshape, blockshape",
    [
        ([100, 100], [50, 50], [20, 20]),
        ([20, 100, 30, 50], [10, 40, 10, 11], [4, 5, 3, 7]),
        ([10, 13, 12, 14, 12, 10], [5, 4, 6, 2, 3, 7], [2, 2, 2, 2, 2, 2]),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "plainbuffer, sequential, filename, filename2",
    [
        (True, False, None, None),
        (False, False, None, None),
        (False, True, None, None),
        (False, True, "test_copy.iarr", "test_copy2.iarr"),
    ],
)
def test_copy(shape, chunkshape, blockshape, dtype, plainbuffer, sequential, filename, filename2):
    if filename and os.path.exists(filename):
        os.remove(filename)
    if filename2 and os.path.exists(filename2):
        os.remove(filename2)

    if plainbuffer:
        storage = ia.Storage(plainbuffer=True)
    else:
        storage = ia.Storage(chunkshape, blockshape, enforce_frame=sequential, filename=filename)
    a_ = ia.linspace(ia.DTShape(shape, dtype), -10, 10, storage=storage)
    sl = tuple([slice(0, s - 1) for s in shape])
    a = a_[sl]
    b = a.copy(filename=filename2)
    an = ia.iarray2numpy(a)
    bn = ia.iarray2numpy(b)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(an, bn, rtol=rtol)

    if filename and os.path.exists(filename):
        os.remove(filename)
    if filename2 and os.path.exists(filename2):
        os.remove(filename2)
