import pytest
import numpy as np
import zarr
import iarray as ia


# linspace
@pytest.mark.parametrize(
    "start, stop, shape, chunks, dtype, contiguous, urlpath",
    [
        (
            0,
            10,
            [100, 120, 50],
            [33, 21, 34],
            'f8',
            False,
            "test_linspace_sparse.ziarr",
        ),
       (-0.1, -0.2, [40, 39, 52, 12], [12, 17, 6, 5], 'float32', True, None),
        (
            0,
            10,
            [55, 24, 31],
            [55, 24, 15],
            'float64',
            True,
            "test_linspace_contiguous.ziarr",
        ),
        (-0.1, -0.2, [4, 3, 5, 2], [4, 3, 5, 2], 'f4', False, None),
    ],
)
def test_linspace(start, stop, shape, chunks, dtype, contiguous, urlpath):
    size = np.prod(shape)
    z = zarr.open('test_linspace.zarr', mode='w', shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.linspace(start, stop, size, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy('test_linspace.zarr', contiguous=contiguous, urlpath=urlpath)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if z.dtype == 'float64' else np.float32
    c = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)

    ia.remove_urlpath(urlpath)


# arange
@pytest.mark.parametrize(
    "start, stop, shape, chunks, dtype, contiguous, urlpath",
    [
        (
            0,
            10,
            [22, 21, 51],
            [12, 14, 22],
            'float64',
            False,
            "test_arange_sparse.ziarr",
        ),
        (
            0,
            1,
            [12, 12, 15, 13, 18, 19],
            [6, 5, 4, 7, 7, 5],
            'float32',
            True,
            None,
        ),
        (
            0,
            10 * 12 * 5,
            [10, 12, 5],
            [5, 5, 5],
            'uint64',
            True,
            "test_arange_contiguous.ziarr",
        ),
        (-0.1, -0.2, [4, 3, 5, 2], [2, 2, 2, 2], 'i4', False, None),
    ],
)
def test_arange(start, stop, shape, chunks, dtype, contiguous, urlpath):
    size = int(np.prod(shape))
    step = (stop - start) / size
    z = zarr.open('test_arange.zarr', mode='w', shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.arange(start, stop, step, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy('test_arange.zarr', contiguous=contiguous, urlpath=urlpath)
    b = ia.iarray2numpy(a)
    npdtype = b.dtype
    c = np.arange(start, stop, step, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)
    ia.remove_urlpath(urlpath)

"""
# from_file
@pytest.mark.parametrize(
    "start, stop, shape, chunks, dtype, contiguous, urlpath",
    [
        (0, 10, [1234], [123], 'float64', False, "test.fromfile0.ziarr"),
        (
            -0.1,
            -0.10,
            [10, 12, 21, 31, 11],
            [4, 3, 5, 5, 2],
            'float32',
            True,
            "test.fromfile1.ziarr",
        ),
    ],
)
def test_from_file(start, stop, shape, chunks, dtype, contiguous, urlpath):
    size = np.prod(shape)
    z = zarr.open('test_linspace.zarr', mode='w', shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.linspace(start, stop, size, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy('test_linspace.zarr', contiguous=contiguous, urlpath=urlpath)

    b = ia.load(urlpath)
    c = ia.iarray2numpy(b)
    npdtype = np.float64 if z.dtype == 'float64' else np.float32
    d = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(c, d)

    ia.remove_urlpath(urlpath)

"""
# get_slice
@pytest.mark.parametrize(
    "start, stop, slice, shape, chunks, dtype, contiguous, urlpath",
    [
        (
            0,
            10,
            (slice(2, 4), slice(5, 10), slice(1, 2)),
            [21, 31, 21],
            [10, 12, 5],
            'float64',
            True,
            None,
        ),
        (
            -0.1,
            -0.2,
            (slice(2, 4), slice(7, 12)),
            [55, 123],
            [12, 16],
            'float32',
            False,
            "test_slice_sparse.ziarr",
        ),
        (
            0,
            10 * 12 * 5,
            (slice(2, 4), slice(5, 10), slice(1, 2)),
            [10, 12, 5],
            [5, 6, 5],
            np.uint32,
            True,
            "test_slice_contiguous.ziarr",
        ),
        (
            -120 * 160,
            0,
            (slice(2, 4), slice(7, 12)),
            [120, 160],
            [120, 40],
            np.int16,
            False,
            None,
        ),
    ],
)
def test_slice(start, stop, slice, shape, chunks, dtype, contiguous, urlpath):
    size = int(np.prod(shape))
    step = (stop - start) / size
    z = zarr.open('test_slice.zarr', mode='w', shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.arange(start, stop, step, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy('test_slice.zarr', contiguous=contiguous, urlpath=urlpath)
    b = a[slice]
    c = ia.iarray2numpy(b)

    npdtype = a.dtype
    d = np.arange(start, stop, step, dtype=npdtype).reshape(shape)[slice]
    if dtype in [np.float64, np.float32]:
        np.testing.assert_allclose(c, d)
    else:
        np.testing.assert_array_equal(c, d)

    ia.remove_urlpath(urlpath)


# zeros
@pytest.mark.parametrize(
    "shape, chunks, dtype, contiguous, urlpath",
    [
        (
            [134, 1234, 238],
            [10, 25, 35],
            np.float64,
            True,
            "test_zeros_contiguous.ziarr",
        ),
        ([456, 431], [102, 16], np.float32, False, "test_zeros_sparse.ziarr"),
        ([10, 12, 5], [10, 1, 1], np.int16, False, None),
        ([12, 16], [1, 16], np.uint8, True, None),
        ([12, 16], [1, 16], 'b', True, "test_zeros_contiguous.ziarr"),
    ],
)
def test_zeros(shape, chunks, dtype, contiguous, urlpath):
    z = zarr.open('test_zeros.zarr', mode='w', shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.zeros(shape=shape, dtype=z.dtype)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy('test_zeros.zarr', contiguous=contiguous, urlpath=urlpath)

    b = ia.iarray2numpy(a)
    npdtype = dtype
    c = np.zeros(shape, dtype=npdtype)
    if dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)

    ia.remove_urlpath(urlpath)


# ones
@pytest.mark.parametrize(
    "shape, chunks, dtype, contiguous, urlpath",
    [
        ([456, 12, 234], [55, 6, 21], np.float64, False, None),
        ([1024, 55], [66, 22], np.float32, True, "test_ones_contiguous.ziarr"),
        ([10, 12, 5], [5, 6, 5], np.int8, True, None),
        ([120, 130], [45, 64], np.uint16, False, "test_ones_sparse.ziarr"),
        ([10, 12, 5], [5, 6, 5], np.bool_, True, None),
    ],
)
def test_ones(shape, chunks, dtype, contiguous, urlpath):
    z = zarr.open('test_ones.zarr', mode='w', shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.ones(shape=shape, dtype=z.dtype)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy('test_ones.zarr', contiguous=contiguous, urlpath=urlpath)

    b = ia.iarray2numpy(a)
    npdtype = dtype
    c = np.ones(shape, dtype=npdtype)
    if dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)

    ia.remove_urlpath(urlpath)


# full
@pytest.mark.parametrize(
    "fill_value, shape, chunks, dtype, contiguous, urlpath",
    [
        (8.34, [123, 432, 222], [24, 31, 15], np.float64, True, None),
        (2.00001, [567, 375], [52, 16], np.float32, False, "test_full_sparse.ziarr"),
        (8, [10, 12, 5], [5, 5, 5], np.int32, True, "test_full_contiguous.ziarr"),
        (True, [12, 16], [12, 16], np.bool_, False, None),
    ],
)
def test_full(fill_value, shape, chunks, dtype, contiguous, urlpath):
    z = zarr.open('test_full.zarr', mode='w', shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.full(shape, fill_value, dtype=z.dtype)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy('test_full.zarr', contiguous=contiguous, urlpath=urlpath)

    b = ia.iarray2numpy(a)
    npdtype = dtype
    c = np.full(shape, fill_value, dtype=npdtype)
    if dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)

    ia.remove_urlpath(urlpath)

"""
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

"""