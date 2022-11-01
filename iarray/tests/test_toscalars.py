import pytest
import iarray as ia
import numpy as np


# Cast to python scalars
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    [
        ([100, 100], [50, 50], [20, 20]),
        ([20, 10, 30], [10, 4, 10], [4, 5, 3]),
        pytest.param(
            [10, 13, 12, 14, 12],
            [5, 4, 6, 2, 3],
            [2, 2, 2, 2, 2],
            marks=pytest.mark.heavy,
        ),
    ],
)
@pytest.mark.parametrize(
    "value, dtype, np_dtype",
    [
        (ia.pi, np.float32, "i4"),
        (2.345554, np.float64, None),
        (123456789, np.int64, "u4"),
        (33, np.int32, "b1"),
        (2**34, np.uint32, None),
    ],
)
@pytest.mark.parametrize(
    "contiguous, urlpath",
    [
        (False, None),
        (True, "test_container.iarr"),
    ],
)
def test_toscalars(shape, chunks, blocks, value, dtype, np_dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)

    a = ia.empty(
        (),
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=contiguous,
        urlpath=urlpath,
    )
    a[...] = value
    out_dtype = np.dtype(dtype) if np_dtype is None else np.dtype(np_dtype)
    an = np.asarray(value, dtype=out_dtype)
    if "f" in out_dtype.str:
        b = float(a)
        bn = float(an)
    elif "i" in out_dtype.str:
        b = int(a)
        bn = int(an)
    else:
        b = bool(a)
        bn = bool(an)

    assert b == bn

    ia.remove_urlpath(urlpath)
