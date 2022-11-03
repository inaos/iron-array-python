import pytest
import numpy as np
import iarray as ia


@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        ([10, 12, 5], [10, 1, 1], [10, 1, 1], np.int16, True, None),
        ([10, 12, 5], [10, 1, 1], [10, 1, 1], np.int16, True, None),
        ([12, 16], [1, 16], [1, 16], np.float32, True, None),
    ],
)
def test_split_concatenate(shape, chunks, blocks, dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    a = ia.arange(
        np.prod(shape),
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=contiguous,
        urlpath=urlpath,
    )
    b = a.split()
    c = ia.concatenate(a.shape, b)

    if "f" in np.dtype(c.dtype).str:
        np.testing.assert_almost_equal(a.data, c.data)
    else:
        np.testing.assert_array_equal(a.data, c.data)

    ia.remove_urlpath(urlpath)
