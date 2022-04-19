import pytest
import iarray as ia
import numpy as np
import zarr

array_data = [
    ([30, 100], [20, 20], [10, 13], [30, 100], 1, True, None),
    ([30, 130], [50, 50], [20, 25], [40, 130], 0, False, None),
    pytest.param([10, 78, 55, 21], [3, 30, 30, 21], [3, 12, 6, 21], [10, 78, 55, 2], 3, True, "test_append_acontiguous.iarr", marks=pytest.mark.heavy),
    ([30, 100], [30, 44], [30, 2], [4, 100], 0, False, "test_append_asparse.iarr"),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.int32, np.uint64, np.uint32])
@pytest.mark.parametrize(
    "shape, chunks, blocks, data_shape, axis, acontiguous, aurlpath",
    array_data,
)
def test_append(shape, chunks, blocks, data_shape, axis, dtype, acontiguous, aurlpath):
    ia.remove_urlpath(aurlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)
    max = 1
    if dtype not in [np.float64, np.float32]:
        for i in range(len(shape)):
            max *= shape[i]
    a = ia.arange(shape, 0, max, cfg=cfg, mode="w", dtype=dtype)

    npa = ia.iarray2numpy(a)
    z = zarr.array(npa, dtype=dtype)

    with pytest.raises(ValueError):
        np_data = np.full(shape=17, fill_value=47, dtype=dtype)
        a.append(np_data, axis=axis)

    np_data = np.full(shape=data_shape, fill_value=47, dtype=dtype)

    a.append(data=np_data, axis=axis)
    z.append(np_data, axis=axis)

    npb = ia.iarray2numpy(a)
    npc = z[:]

    if dtype in [np.float64, np.float32]:
        rtol = 1e-6 if dtype == np.float32 else 1e-14
        np.testing.assert_allclose(npb, npc, rtol=rtol, atol=0)
    else:
        np.testing.assert_equal(npb, npc)

    ia.remove_urlpath(aurlpath)