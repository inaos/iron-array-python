import pytest
import iarray as ia
import numpy as np
import zarr

array_data = [
    ([30, 100], [20, 20], [10, 13], [30, 100], 0, 30, True, None),
    ([30, 130], [50, 50], [20, 25], [30, 100], 1, 50, False, None),
    pytest.param([10, 78, 55, 21], [3, 30, 30, 21], [3, 12, 6, 21], [10, 78, 55, 42], 3, 0, True, "test_insert_acontiguous.iarr", marks=pytest.mark.heavy),
    ([30, 100], [30, 44], [30, 2], [30, 100], 0, 0, False, "test_insert_asparse.iarr"),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.int32, np.uint64, np.uint32])
@pytest.mark.parametrize(
    "shape, chunks, blocks, data_shape, axis, start, acontiguous, aurlpath",
    array_data,
)
def test_insert(shape, chunks, blocks, data_shape, axis, start, dtype, acontiguous, aurlpath):
    ia.remove_urlpath(aurlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)
    max = 1
    if dtype not in [np.float64, np.float32]:
        for i in range(len(shape)):
            max *= shape[i]
    a = ia.arange(shape, 0, max, cfg=cfg, mode="w", dtype=dtype)

    with pytest.raises(ValueError):
        np_data = np.full(shape=5, fill_value=47, dtype=dtype)
        a.insert(data=np_data, axis=axis, start=0)
    with pytest.raises(IndexError):
        np_data = np.full(shape=data_shape, fill_value=47, dtype=dtype)
        a.insert(start=1, data=np_data, axis=axis)

    np_data = np.full(shape=data_shape, fill_value=47, dtype=dtype)
    a.insert(data=np_data, axis=axis, start=start)

    slice_ = []
    for i in range(0, a.ndim):
        if i != axis:
            slice_.append(slice(0, shape[i]))
        else:
            slice_.append(slice(start, start + data_shape[i]))

    npa = ia.iarray2numpy(a)
    if dtype in [np.float64, np.float32]:
        rtol = 1e-6 if dtype == np.float32 else 1e-14
        np.testing.assert_allclose(npa[slice_], np_data, rtol=rtol, atol=0)
    else:
        np.testing.assert_equal(npa[slice_], np_data)

    ia.remove_urlpath(aurlpath)
