import pytest
import iarray as ia
import numpy as np
from math import isclose


# Slice

slice_data = [
    ([30, 100], [20, 20], [10, 13], True, None),
    ([30, 130], [50, 50], [20, 25], False, None),
    pytest.param(
        [10, 78, 55, 21],
        [3, 30, 30, 21],
        [3, 12, 6, 21],
        True,
        "test_slice_acontiguous.iarr",
        marks=pytest.mark.heavy,
    ),
    ([30, 100], [30, 44], [30, 2], False, "test_slice_asparse.iarr"),
]


@pytest.mark.parametrize(
    "slices",
    [
        (0, 1),  # test with scalars
        slice(10, 20),
        9,
        (slice(5, 30), slice(10, 40)),
        (slice(5, 5), slice(5, 23)),
        (slice(5, 5), slice(3, 12)),
        (slice(5, 30), 47, ...),
        (..., slice(5, 6)),
    ],
)
@pytest.mark.parametrize(
    "dtype, np_dtype",
    [
        (np.float32, "f2"),
        (np.uint64, ">M8[ms]"),
        (np.int64, None),
        (np.int32, None),
        (np.uint64, ">i4"),
        (np.uint32, "u8"),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, acontiguous, aurlpath",
    slice_data,
)
def test_slice(slices, shape, chunks, blocks, dtype, np_dtype, acontiguous, aurlpath):
    ia.remove_urlpath(aurlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)
    max = 1
    out_dtype = dtype if np_dtype is None else np.dtype(np_dtype)
    if out_dtype not in [np.float64, np.float32]:
        for i in range(len(shape)):
            max *= shape[i]
    a = ia.arange(shape, 0, max, cfg=cfg, mode="w", dtype=dtype, np_dtype=np_dtype)
    an = ia.iarray2numpy(a)

    a[slices] = 0
    an[slices] = 0
    if out_dtype in [np.float16, np.float32, np.float64]:
        np.testing.assert_almost_equal(a.data, an)
    else:
        np.testing.assert_equal(a.data, an)

    data = ia.arange(shape, dtype=dtype, np_dtype=np_dtype)[slices]

    a[slices] = data
    an[slices] = data.data if isinstance(data, ia.IArray) else data
    if out_dtype in [np.float16, np.float32, np.float64]:
        np.testing.assert_almost_equal(a.data, an)
    else:
        np.testing.assert_equal(a.data, an)

    b = a[slices]
    an2 = an[slices]

    if b.ndim == 0:
        if out_dtype in [np.float16, np.float32, np.float64]:
            isclose(an2, b)
        else:
            assert an2 == b
    else:
        bn = ia.iarray2numpy(b)
        assert an2.shape == bn.shape
        assert an2.ndim == bn.ndim
        if out_dtype in [np.float16, np.float32, np.float64]:
            np.testing.assert_almost_equal(an[slices], bn)
        else:
            np.testing.assert_equal(an[slices], bn)

    ia.remove_urlpath(aurlpath)


@pytest.mark.parametrize(
    "dtype, np_dtype",
    [
        (np.float32, "f2"),
        (np.float64, ">f4"),
        (np.float64, None),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, acontiguous, aurlpath",
    slice_data,
)
def test_double_slice(shape, chunks, blocks, dtype, np_dtype, acontiguous, aurlpath):
    ia.remove_urlpath(aurlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)

    a = ia.linspace(shape, -10, 10, cfg=cfg, mode="a", dtype=dtype, np_dtype=np_dtype)
    b1 = a[4]
    b2 = a[4]
    np.testing.assert_almost_equal(b1.data, b2.data)

    ia.remove_urlpath(aurlpath)
