import pytest
import iarray as ia
import numpy as np
from math import isclose


# Slice


@pytest.mark.parametrize(
    "slices",
    [
        slice(10, 20),
        60,
        (slice(5, 30), slice(10, 40)),
        (slice(5, 30), 47, ...),
        (..., slice(5, 6)),
        (slice(5, 10), slice(5, 5)),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape, chunkshape, blockshape",
    [
        # ([200], [70], [30], np.float64),
        ([100, 100], [20, 20], [10, 13]),
        ([100, 130], None, None),
        ([98, 78, 55, 21], None, None),
        ([100, 100], None, None),
    ],
)
def test_slice(slices, shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.Storage(plainbuffer=True)
    else:
        storage = ia.Storage(chunkshape, blockshape, enforce_frame=True)

    a = ia.linspace(ia.DTShape(shape, dtype), -10, 10, storage=storage)
    an = ia.iarray2numpy(a)

    b = a[slices]

    if b.ndim == 0:
        isclose(an[slices], b)
    else:
        bn = ia.iarray2numpy(b)
        np.testing.assert_almost_equal(an[slices], bn)
