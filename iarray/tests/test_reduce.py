import pytest
import iarray as ia
import numpy as np
from math import isclose


params_names = "shape, chunkshape, blockshape, axis, dtype"
params_data = [
    ([100, 100], [50, 50], [20, 20], 0, np.float32),
    ([20, 100, 30, 50], [10, 40, 10, 11], [4, 5, 3, 7], 1, np.float64),
    ([10, 10], [4, 4], [2, 2], 1, np.float64),
    ([70, 45, 56, 34], [20, 23, 30, 34], [9, 7, 8, 7], (0, 3), np.float32),
    ([10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], None, np.float64),
]


@pytest.mark.parametrize(params_names, params_data)
@pytest.mark.parametrize("rfunc", ["mean", "sum", "prod", "max", "min"])
def test_reduce(shape, chunkshape, blockshape, axis, dtype, rfunc):

    storage = ia.Storage(chunkshape, blockshape)
    a1 = ia.linspace(ia.DTShape(shape, dtype), -1, 0, storage=storage)
    a2 = ia.iarray2numpy(a1)

    b2 = getattr(np, rfunc)(a2, axis=axis)
    b1 = getattr(ia, rfunc)(a1, axis=axis)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    if b2.ndim == 0:
        isclose(b1, b2, rel_tol=rtol, abs_tol=0.0)
    else:
        np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol, atol=0)
