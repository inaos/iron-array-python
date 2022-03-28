import pytest
import iarray as ia
import numpy as np
from math import isclose

params_names = "shape, chunks, blocks, axis, dtype, mode"
params_data = [
    ([100, 100], [50, 50], [20, 20], 0, np.float32, "r"),
    ([10, 10, 10, 10], [4, 4, 4, 4], [2, 2, 2, 2], 1, np.float64, "r+"),
    pytest.param([20, 100, 30, 50], [10, 40, 10, 11], [4, 5, 3, 7], 1, np.float64, "r+", marks=pytest.mark.heavy),
    ([10, 10, 10, 10], [4, 4, 4, 4], [2, 2, 2, 2], 1, np.int16, "w"),
    ([40, 45], [20, 23], [9, 7], (0), np.int64, "w-"),
    pytest.param([70, 45, 56, 34], [20, 23, 30, 34], [9, 7, 8, 7], (0, 3), np.int64, "w-", marks=pytest.mark.heavy),
    ([10, 10, 10], [10, 10, 10], [10, 10, 10], None, np.uint32, "a"),
    ([10, 10, 10], [4, 4, 4], [2, 2, 2], 1, np.bool_, "w"),
]


@pytest.mark.parametrize(params_names, params_data)
@pytest.mark.parametrize("rfunc", ["mean", "sum", "prod", "max", "min"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "test_reduce.iarr"])
@pytest.mark.parametrize("view", [True, False])
def test_reduce(shape, chunks, blocks, axis, dtype, rfunc, contiguous, urlpath, mode, view):

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=contiguous, urlpath=urlpath)
    a1 = ia.ones(shape, dtype=dtype, cfg=cfg)
    if view:
        slices = tuple([slice(np.random.randint(1, s)) for s in a1.shape])
        a1 = a1[slices]

    a2 = a1.data

    b2 = getattr(np, rfunc)(a2, axis=axis)
    if mode in ["r", "r+"]:
        with pytest.raises(IOError):
            b1 = getattr(ia, rfunc)(a1, axis=axis, urlpath="test_reduce_res.iarr", mode=mode)
        mode = "a"
    b1 = getattr(ia, rfunc)(a1, axis=axis, urlpath="test_reduce_res.iarr", mode=mode)

    if dtype in [np.float64, np.float32] or rfunc == "mean":
        rtol = 1e-6 if dtype == np.float32 else 1e-14
        if b2.ndim == 0:
            isclose(b1, b2, rel_tol=rtol, abs_tol=0.0)
        else:
            np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol, atol=0)
    else:
        if b2.ndim == 0:
            assert b1 == b2
        else:
            np.testing.assert_array_equal(ia.iarray2numpy(b1), b2)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_reduce_res.iarr")
