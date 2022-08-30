import pytest
import numpy as np
import iarray as ia


# linspace
@pytest.mark.parametrize(
    "start, stop, shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath",
    [
        pytest.param(
            0,
            10,
            [100, 90, 50],
            [33, 21, 34],
            [12, 13, 7],
            np.float64,
            np.dtype("f8"),
            False,
            "test_linspace_sparse.iarr",
            marks=pytest.mark.heavy,
        ),
        pytest.param(
            -0.1,
            -0.2,
            [40, 39, 52, 12],
            [12, 17, 6, 5],
            [5, 4, 6, 5],
            np.float32,
            ">f4",
            True,
            None,
            marks=pytest.mark.heavy,
        ),
        (
            0,
            10,
            [55, 24, 31],
            [55, 24, 15],
            [55, 24, 5],
            np.float64,
            None,
            True,
            "test_linspace_contiguous.iarr",
        ),
        (-0.1, -0.2, [4, 3, 5, 2], [4, 3, 5, 2], [2, 3, 2, 2], np.float32, "M8[D]", False, None),
    ],
)
def test_linspace(start, stop, shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath):
    size = int(np.prod(shape))
    ia.remove_urlpath(urlpath)
    a = ia.linspace(
        shape,
        start,
        stop,
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=contiguous,
        urlpath=urlpath,
    )
    b = ia.iarray2numpy(a)
    c = np.linspace(start, stop, size, dtype=np_dtype).reshape(shape)
    if c.dtype in [np.float32, np.float64]:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_equal(b, c)

    ia.remove_urlpath(urlpath)


# arange
@pytest.mark.parametrize(
    "start, stop, shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath",
    [
        (
            0,
            10,
            [22, 21, 51],
            [12, 14, 22],
            [5, 3, 6],
            np.float64,
            np.dtype(">f8"),
            False,
            "test_arange_sparse.iarr",
        ),
        pytest.param(
            0,
            1,
            [12, 12, 15, 13, 18, 19],
            [6, 5, 4, 7, 7, 5],
            [2, 2, 1, 2, 3, 3],
            np.float32,
            None,
            True,
            None,
            marks=pytest.mark.heavy,
        ),
        (
            0,
            10 * 12 * 5,
            [10, 12, 5],
            [5, 5, 5],
            [2, 1, 2],
            np.uint64,
            None,
            True,
            "test_arange_contiguous.iarr",
        ),
        (-0.1, -0.2, [4, 3, 5, 2], [2, 2, 2, 2], [2, 2, 2, 2], np.int32, None, False, None),
        (
            np.datetime64("1964", "D"),
            np.datetime64("1965-04", "D"),
            [228, 2],
            [100, 1],
            [40, 1],
            np.int64,
            "<M8[D]",
            True,
            "test_arange_contiguous.iarr",
        ),
        (
            np.timedelta64(1234, "Y"),
            np.timedelta64(11234, "Y"),
            [100, 100],
            [54, 50],
            [25, 20],
            np.int64,
            "<m8[Y]",
            True,
            "test_arange_contiguous.iarr",
        ),
    ],
)
def test_arange(start, stop, shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath):
    size = int(np.prod(shape))
    if type(start) in [np.datetime64, np.timedelta64]:
        step = 1
    else:
        step = (stop - start) / size
    ia.remove_urlpath(urlpath)
    a = ia.arange(
        shape,
        start,
        stop,
        step,
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=contiguous,
        urlpath=urlpath,
    )
    b = ia.iarray2numpy(a)
    out_dtype = dtype if np_dtype is None else np_dtype
    c = np.arange(start, stop, step, dtype=out_dtype).reshape(shape)
    if "f" in np.dtype(out_dtype).str:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)
    ia.remove_urlpath(urlpath)


# from_file
@pytest.mark.parametrize(
    "start, stop, shape, chunks, blocks, np_dtype, contiguous, urlpath",
    [
        (0, 10, [1234], [123], [21], "f2", False, "test.fromfile0.iarr"),
        (
            -0.1,
            -0.10,
            [10, 12, 21],
            [4, 3, 5],
            [2, 3, 2],
            b"f8",
            True,
            "test.fromfile1.iarr",
        ),
        pytest.param(
            -0.1,
            -0.10,
            [10, 12, 21, 31, 11],
            [4, 3, 5, 5, 2],
            [2, 3, 2, 3, 2],
            np.dtype("f4"),
            True,
            "test.fromfile2.iarr",
            marks=pytest.mark.heavy,
        ),
    ],
)
def test_from_file(start, stop, shape, chunks, blocks, np_dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    size = int(np.prod(shape))
    a = np.linspace(start, stop, size, dtype=np_dtype).reshape(shape)
    ia.numpy2iarray(a, chunks=chunks, blocks=blocks, contiguous=contiguous, urlpath=urlpath)
    c = ia.load(urlpath)
    d = ia.iarray2numpy(c)
    np.testing.assert_almost_equal(a, d)
    ia.remove_urlpath(urlpath)


# get_slice
@pytest.mark.parametrize(
    "start, stop, slice, shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath",
    [
        (
            0,
            10,
            (slice(2, 4), slice(5, 10), slice(1, 2)),
            [21, 31, 21],
            [10, 12, 5],
            [3, 6, 2],
            np.float64,
            None,
            True,
            None,
        ),
        (
            -0.1,
            -0.2,
            (slice(2, 4), slice(7, 12)),
            [55, 123],
            [12, 16],
            [5, 7],
            np.float64,
            "f4",
            False,
            "test_slice_sparse.iarr",
        ),
        (
            0,
            10 * 12 * 5,
            (slice(2, 4), slice(5, 10), slice(1, 2)),
            [10, 12, 5],
            [5, 6, 5],
            [5, 6, 5],
            np.uint32,
            "i8",
            True,
            "test_slice_contiguous.iarr",
        ),
        (
            -120 * 160,
            0,
            (slice(2, 4), slice(7, 12)),
            [120, 160],
            [120, 40],
            [30, 20],
            np.int16,
            None,
            False,
            None,
        ),
        (
            np.datetime64("1993", "M"),
            np.datetime64("1999", "M"),
            (slice(2, 4), slice(1, 2)),
            [8, 9],
            [3, 4],
            [3, 2],
            np.dtype(np.float64),
            "M8[M]",
            True,
            None,
        ),
    ],
)
def test_slice(start, stop, slice, shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath):
    size = int(np.prod(shape))
    if type(start) in [np.datetime64, np.timedelta64]:
        step = 1
    else:
        step = (stop - start) / size
    ia.remove_urlpath(urlpath)
    a = ia.arange(
        shape,
        start,
        stop,
        step,
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=contiguous,
        urlpath=urlpath,
    )
    b = a[slice]
    c = ia.iarray2numpy(b)
    out_dtype = dtype if np_dtype is None else np_dtype
    d = np.arange(start, stop, step, dtype=out_dtype).reshape(shape)[slice]
    if "f" in np.dtype(out_dtype).str:
        np.testing.assert_allclose(c, d)
    else:
        np.testing.assert_array_equal(c, d)
    ia.remove_urlpath(urlpath)


# empty  # TODO: make it work properly
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath",
    [
        ([55, 123, 72], [10, 12, 25], [2, 3, 7], np.dtype(np.float64), ">M8[h]", True, None),
        ([10, 12, 5], [10, 12, 1], [5, 12, 1], np.float32, "i8", False, "test_empty_sparse.iarr"),
        (
            [55, 123, 72],
            [10, 12, 25],
            [2, 3, 7],
            np.bool_,
            "u1",
            True,
            "test_empty_contiguous.iarr",
        ),
        ([10, 12, 5], [5, 12, 5], [2, 12, 5], np.int64, np.dtype("m8[ns]"), False, None),
    ],
)
def test_empty(shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    with ia.config(chunks=chunks, blocks=blocks, contiguous=contiguous, urlpath=urlpath):
        a = ia.empty(shape, dtype=dtype, np_dtype=np_dtype)
    b = ia.iarray2numpy(a)
    out_dtype = dtype if np_dtype is None else np_dtype
    assert b.dtype == np.dtype(out_dtype)
    assert b.shape == tuple(shape)

    ia.remove_urlpath(urlpath)


# zeros
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath",
    [
        pytest.param(
            [134, 1234, 238],
            [10, 25, 35],
            [2, 7, 12],
            np.float64,
            None,
            True,
            "test_zeros_contiguous.iarr",
            marks=pytest.mark.heavy,
        ),
        ([456, 431], [102, 16], [12, 7], np.float32, None, False, "test_zeros_sparse.iarr"),
        ([10, 12, 5], [10, 1, 1], [10, 1, 1], np.int16, None, False, None),
        ([10, 12, 5], [10, 1, 1], [10, 1, 1], np.int16, "|b1", False, None),
        ([12, 16], [1, 16], [1, 16], np.float32, ">M8[Y]", True, None),
        ([12, 16], [1, 16], [1, 16], np.bool_, ">i8", True, "test_zeros_contiguous.iarr"),
    ],
)
def test_zeros(shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    a = ia.zeros(
        shape,
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=contiguous,
        urlpath=urlpath,
    )
    b = ia.iarray2numpy(a)
    out_dtype = dtype if np_dtype is None else np_dtype
    c = np.zeros(shape, dtype=out_dtype)
    if "f" in np.dtype(out_dtype).str:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)

    ia.remove_urlpath(urlpath)


# ones
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath",
    [
        ([100, 12, 34], [55, 6, 21], [12, 3, 5], np.float64, None, False, None),
        pytest.param(
            [456, 12, 234],
            [55, 6, 21],
            [12, 3, 5],
            np.float64,
            None,
            False,
            None,
            marks=pytest.mark.heavy,
        ),
        ([100, 55], [66, 22], [12, 3], np.float32, None, True, "test_ones_contiguous.iarr"),
        ([10, 12, 5], [5, 6, 5], [5, 3, 5], np.int8, None, True, None),
        ([120, 130], [45, 64], [33, 12], np.uint16, None, False, "test_ones_sparse.iarr"),
        ([10, 12, 5], [5, 6, 5], [5, 3, 5], np.bool_, None, True, None),
        ([12, 5], [5, 5], [3, 5], np.float32, ">M8[Y]", True, None),
        ([10, 12], [5, 6], [5, 3], np.int8, ">m8[ps]", False, None),
    ],
)
def test_ones(shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    a = ia.ones(
        shape,
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=contiguous,
        urlpath=urlpath,
    )
    b = ia.iarray2numpy(a)
    out_dtype = dtype if np_dtype is None else np_dtype
    c = np.ones(shape, dtype=out_dtype)
    if "f" in np.dtype(out_dtype).str:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)

    ia.remove_urlpath(urlpath)


# full
@pytest.mark.parametrize(
    "fill_value, shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath",
    [
        pytest.param(
            8.34,
            [123, 432, 222],
            [24, 31, 15],
            [6, 6, 6],
            np.dtype(np.float64),
            None,
            True,
            None,
            marks=pytest.mark.heavy,
        ),
        (2.00001, [56, 37], [20, 16], [9, 7], np.float32, None, False, "test_full_sparse.iarr"),
        pytest.param(
            2.00001,
            [567, 375],
            [52, 16],
            [9, 7],
            np.float32,
            None,
            False,
            "test_full_sparse.iarr",
            marks=pytest.mark.heavy,
        ),
        (8, [10, 12, 5], [5, 5, 5], [5, 5, 5], np.int32, None, True, "test_full_contiguous.iarr"),
        (True, [12, 16], [12, 16], [10, 10], np.uint8, np.dtype(np.bool_).str, False, None),
        (
            np.datetime64("1929", "D"),
            [12, 16],
            [12, 16],
            [10, 10],
            np.int32,
            "<M8[D]",
            False,
            None,
        ),
        (
            np.datetime64("1929", "ns"),
            [10, 12, 5],
            [5, 5, 5],
            [5, 5, 5],
            np.int64,
            np.dtype("datetime64[ns]").str,
            True,
            "test_full_contiguous.iarr",
        ),
        (
            np.datetime64("2034", "Y"),
            [56, 37],
            [20, 16],
            [9, 7],
            np.uint16,
            ">M8[Y]",
            False,
            "test_full_sparse.iarr",
        ),
        (
            np.timedelta64(2034, "h"),
            [50, 42],
            [20, 16],
            [9, 7],
            np.uint32,
            ">m8[h]",
            False,
            "test_full_sparse.iarr",
        ),
        (4, [12, 16], [12, 16], [10, 10], np.uint8, np.dtype(np.float64).str, True, None),
        (0.000056, [12, 16], [12, 16], [10, 10], np.double, ">f8", False, None),
    ],
)
def test_full(fill_value, shape, chunks, blocks, dtype, np_dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    a = ia.full(
        shape,
        fill_value,
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        contiguous=contiguous,
        urlpath=urlpath,
    )
    b = ia.iarray2numpy(a)
    out_dtype = dtype if np_dtype is None else np_dtype
    c = np.full(shape, fill_value, dtype=out_dtype)
    if "f" in np.dtype(out_dtype).str:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)

    ia.remove_urlpath(urlpath)


# TODO: Update this when persistent sparse would be supported
@pytest.mark.parametrize(
    "contiguous",
    [
        (True,),
        (False,),
    ],
)
def test_overwrite(contiguous):
    fname = "pepe.iarr"
    ia.remove_urlpath(fname)
    a = ia.arange([10, 20, 10, 14], contiguous=contiguous, urlpath=fname)
    b = ia.arange([10, 20, 10, 14], contiguous=contiguous, urlpath=fname, mode="w")
    with pytest.raises(IOError):
        b = ia.arange([10, 20, 10, 14], contiguous=contiguous, urlpath=fname, mode="w-")

    ia.remove_urlpath(fname)


# numpy views
@pytest.mark.parametrize(
    "shape, starts, stops, np_dtype, contiguous, urlpath",
    [
        pytest.param(
            [55, 123, 72],
            [10, 12, 25],
            [12, 14, 26],
            np.float64,
            False,
            "test_view_sparse.iarr",
            marks=pytest.mark.heavy,
        ),
        ([55, 123, 72], [10, 12, 25], [12, 14, 26], np.float64, True, "test_view_contiguous.iarr"),
        ([10, 12, 5], [3, 9, 1], [4, 12, 3], "f8", True, None),
        ([10, 12, 5], [3, 9, 1], [4, 12, 3], "i2", False, None),
    ],
)
def test_view(shape, starts, stops, np_dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    cfg = ia.set_config_defaults(contiguous=contiguous, urlpath=urlpath)
    nelems = np.prod(shape)
    a = np.linspace(0, 1, nelems, dtype=np_dtype).reshape(shape)
    slice_ = tuple(slice(i, j) for i, j in zip(starts, stops))
    a_view = a[slice_]
    b = ia.numpy2iarray(a_view, cfg=cfg)
    dtype = b.dtype if b.np_dtype is None else b.np_dtype
    assert dtype == a.dtype
    assert b.shape == a_view.shape
    np.testing.assert_almost_equal(a_view, b.data)

    ia.remove_urlpath(urlpath)


# numpy arrays stored in Fortran ordering
@pytest.mark.parametrize(
    "shape, np_dtype, contiguous, urlpath",
    [
        ([55, 13, 2], np.float64, True, None),
        ([55, 13, 2], np.int16, False, None),
        ([10, 12], np.float32, False, "test_fortran_sparse.iarr"),
        ([10, 12], "m8[h]", True, "test_fortran_contiguous.iarr"),
    ],
)
def test_fortran(shape, np_dtype, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    cfg = ia.set_config_defaults(contiguous=contiguous, urlpath=urlpath)
    nelems = np.prod(shape)
    a = np.linspace(0, 1, nelems, dtype=np_dtype).reshape(shape)
    a_fortran = a.copy(order="F")
    b = ia.numpy2iarray(a_fortran, cfg=cfg)
    dtype = b.dtype if b.np_dtype is None else b.np_dtype
    assert dtype == a.dtype
    assert b.shape == a.shape
    assert b.shape == a_fortran.shape
    if dtype in [np.float32, np.float64]:
        np.testing.assert_almost_equal(a_fortran, b.data)
    else:
        np.testing.assert_equal(a_fortran, b.data)

    ia.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    "clevel, codec, filters",
    [
        (1, ia.Codec.ZSTD, [ia.Filter.NOFILTER]),
    ],
)
def test_btune_copy(clevel, codec, filters):
    a = ia.linspace([100], 0, 1)
    a_copy = a.copy(clevel=clevel, codec=codec, filters=filters, btune=False)
    np.testing.assert_almost_equal(a_copy.data, a.data)
