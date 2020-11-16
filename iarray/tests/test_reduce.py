import pytest
import iarray as ia
import numpy as np


params_names = "shape, chunkshape, blockshape, axis, dtype"
params_data = [
    # ([100, 100], [50, 50], [20, 20], 0, np.float32),
    # ([20, 100, 30, 50], [10, 40, 10, 11], [4, 5, 3, 7], 1, np.float64),
    ([10, 10], [4, 4], [2, 2], 1, np.float64),
]


@pytest.mark.parametrize(params_names, params_data)
@pytest.mark.parametrize("rfunc", ["mean"])
@pytest.mark.parametrize("iafunc", ["reduce", "reduce2"])
def test_reduce(shape, chunkshape, blockshape, axis, dtype, rfunc, iafunc):

    storage = ia.Storage(chunkshape, blockshape)
    a1 = ia.arange(ia.DTShape(shape, dtype), storage=storage)
    a2 = ia.iarray2numpy(a1)

    b2 = getattr(np, rfunc)(a2, axis=axis)
    iafunc = getattr(ia, iafunc)
    rfunc = getattr(ia.Reduce, rfunc.upper())
    b1 = iafunc(a1, method=rfunc, axis=axis)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol)
