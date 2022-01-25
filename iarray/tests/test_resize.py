import pytest
import iarray as ia
import numpy as np

array_data = [
    ([30, 100], [20, 20], [10, 13], [30, 100], True, None),
    ([30, 130], [50, 50], [20, 25], [40, 130], False, None),
    ([10, 78, 55, 21], [3, 30, 30, 21], [3, 12, 6, 21], [11, 80, 60, 30], True, "test_resize_acontiguous.iarr"),
    ([30, 100], [30, 44], [30, 2], [30, 150], False, "test_resize_asparse.iarr"),
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
    a.resize(newshape)
    assert a.shape == tuple(newshape)

    ia.remove_urlpath(aurlpath)
