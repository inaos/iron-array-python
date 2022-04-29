import pytest
import iarray as ia
import numpy as np
import zarr

array_data = [
    ([30, 100], [20, 20], [10, 13], 0, 0, 20, True, None),
    ([30, 130], [50, 50], [20, 25], 1, 70, 60, False, None),
    pytest.param([10, 78, 55, 21], [3, 30, 30, 21], [3, 12, 6, 21], 3, 4, 17, True, "test_delete_acontiguous.iarr", marks=pytest.mark.heavy),
    ([30, 100], [30, 44], [30, 2],  1, 44, 44, False, "test_delete_asparse.iarr"),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.int32, np.uint64, np.uint32])
@pytest.mark.parametrize(
    "shape, chunks, blocks, axis, start, delete_len, acontiguous, aurlpath",
    array_data,
)
def test_delete(shape, chunks, blocks, axis, start, delete_len, dtype, acontiguous, aurlpath):
    ia.remove_urlpath(aurlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)
    a = ia.full(shape=shape, fill_value=4, cfg=cfg, mode="w", dtype=dtype)

    with pytest.raises(IndexError):
        a.delete(start=0, delete_len=(chunks[axis]-1), axis=axis)

    slice_ = []
    for i in range(0, a.ndim):
        if i != axis:
            slice_.append(slice(0, shape[i]))
        else:
            slice_.append(slice(start, start + delete_len))
    print(slice_)
    slice_ = tuple(slice_)
    a[slice_] = 0
    a.delete(axis=axis, start=start, delete_len=delete_len)
    npa = ia.iarray2numpy(a)
    npb = np.full(shape=a.shape, fill_value=4, dtype=dtype)

    if dtype in [np.float64, np.float32]:
        rtol = 1e-6 if dtype == np.float32 else 1e-14
        np.testing.assert_allclose(npa, npb, rtol=rtol, atol=0)
    else:
        np.testing.assert_equal(npa, npb)

    ia.remove_urlpath(aurlpath)
