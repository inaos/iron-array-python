import pytest
import iarray as ia
import numpy as np

array_data = [
    ([30, 100], [20, 20], [10, 13], [30, 100], True, None),
    ([30, 130], [50, 50], [20, 25], [40, 130], False, None),
    pytest.param([10, 78, 55, 21], [3, 30, 30, 21], [3, 12, 6, 21], [5, 80, 10, 21], True, "test_resize_acontiguous.iarr", marks=pytest.mark.heavy),
    ([30, 100], [30, 44], [30, 2], [30, 10], False, "test_resize_asparse.iarr"),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.int32, np.uint64, np.uint32])
@pytest.mark.parametrize(
    "shape, chunks, blocks, newshape, acontiguous, aurlpath",
    array_data,
)
def test_resize(shape, chunks, blocks, newshape, dtype, acontiguous, aurlpath):
    ia.remove_urlpath(aurlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)
    max = 1
    if dtype not in [np.float64, np.float32]:
        for i in range(len(shape)):
            max *= shape[i]
    a = ia.arange(shape, 0, max, cfg=cfg, mode="w", dtype=dtype)
    npa = ia.iarray2numpy(a)
    a.resize(newshape)
    assert a.shape == tuple(newshape)

    data_shape = []
    for i in range(len(shape)):
        if shape[i] >= newshape[i]:
            data_shape.append(newshape[i])
        else:
            data_shape.append(shape[i])

    npb = ia.iarray2numpy(a)
    slice_ = tuple(slice(0, i) for i in data_shape)
    if dtype in [np.float64, np.float32]:
        rtol = 1e-6 if dtype == np.float32 else 1e-14
        np.testing.assert_allclose(npa[slice_], npb[slice_], rtol=rtol, atol=0)
    else:
        np.testing.assert_equal(npa[slice_], npb[slice_])

    ia.remove_urlpath(aurlpath)
