import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "shape, dtype, dtype_compat",
    [
        ([200], np.int8, np.uint8),
        ([100, 100], np.int8, np.bool_),
    ],
)
def test_compat_dtype(shape, dtype, dtype_compat):
    a = ia.arange(shape, dtype=dtype)
    b = a.astype(dtype_compat)
    assert a.chunks == b.chunks
    assert a.blocks == b.blocks
    bnp = a.data.astype(dtype_compat)
    np.testing.assert_array_equal(b.data, bnp)
