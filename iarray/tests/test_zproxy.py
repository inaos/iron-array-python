import pytest
import numpy as np
import s3fs
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
            "test_linspace_sparse.iarr",
        ),
        (-0.1, -0.2, [40, 39, 52, 12], [12, 17, 6, 5], 'float32', True, None),
        (
            0,
            10,
            [55, 24, 31],
            [55, 24, 15],
            'float64',
            True,
            "test_linspace_contiguous.iarr",
        ),
        (-0.1, -0.2, [4, 3, 5, 2], [4, 3, 5, 2], 'f4', False, None),
    ],
)
def test_linspace_zproxy(start, stop, shape, chunks, dtype, contiguous, urlpath):
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
    ia.remove_urlpath('test_linspace.zarr')


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
            "test_arange_sparse.iarr",
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
            "test_arange_contiguous.iarr",
        ),
        (-0.1, -0.2, [4, 3, 5, 2], [2, 2, 2, 2], 'i4', False, None),
    ],
)
def test_arange_zproxy(start, stop, shape, chunks, dtype, contiguous, urlpath):
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
    ia.remove_urlpath('test_arange.zarr')


# from_file
@pytest.mark.parametrize(
    "start, stop, shape, chunks, dtype, contiguous, urlpath",
    [
        (0, 10, [1234], [123], 'float64', False, "test.fromfile0.iarr"),
        (
            -0.1,
            -0.10,
            [10, 12, 21, 31, 11],
            [4, 3, 5, 5, 2],
            'float32',
            True,
            "test.fromfile1.iarr",
        ),
    ],
)
def test_from_file_zproxy(start, stop, shape, chunks, dtype, contiguous, urlpath):
    size = np.prod(shape)
    z = zarr.open('test_linspace.zarr', mode='w', shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.linspace(start, stop, size, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    ia.zarr_proxy('test_linspace.zarr', contiguous=contiguous, urlpath=urlpath)

    b = ia.load(urlpath)
    c = ia.iarray2numpy(b)
    npdtype = np.float64 if z.dtype == 'float64' else np.float32
    d = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(c, d)

    e = ia.open(urlpath)
    f = ia.iarray2numpy(e)
    np.testing.assert_almost_equal(f, d)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath('test_linspace.zarr')


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
            "test_slice_sparse.iarr",
        ),
        (
            0,
            10 * 12 * 5,
            (slice(2, 4), slice(5, 10), slice(1, 2)),
            [10, 12, 5],
            [5, 6, 5],
            np.uint32,
            True,
            "test_slice_contiguous.iarr",
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
def test_slice_zproxy(start, stop, slice, shape, chunks, dtype, contiguous, urlpath):
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
    ia.remove_urlpath('test_slice.zarr')


# zeros
@pytest.mark.parametrize(
    "shape, chunks, dtype, contiguous, urlpath",
    [
        (
            [134, 1234, 238],
            [10, 25, 35],
            np.float64,
            True,
            "test_zeros_contiguous.iarr",
        ),
        ([456, 431], [102, 16], np.float32, False, "test_zeros_sparse.iarr"),
        ([10, 12, 5], [10, 1, 1], np.int16, False, None),
        ([12, 16], [1, 16], np.uint8, True, None),
        ([12, 16], [1, 16], 'b', True, "test_zeros_contiguous.iarr"),
    ],
)
def test_zeros_zproxy(shape, chunks, dtype, contiguous, urlpath):
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
    ia.remove_urlpath('test_zeros.zarr')


# ones
@pytest.mark.parametrize(
    "shape, chunks, dtype, contiguous, urlpath",
    [
        ([456, 12, 234], [55, 6, 21], np.float64, False, None),
        ([1024, 55], [66, 22], np.float32, True, "test_ones_contiguous.iarr"),
        ([10, 12, 5], [5, 6, 5], np.int8, True, None),
        ([120, 130], [45, 64], np.uint16, False, "test_ones_sparse.iarr"),
        ([10, 12, 5], [5, 6, 5], np.bool_, True, None),
    ],
)
def test_ones_zproxy(shape, chunks, dtype, contiguous, urlpath):
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
    ia.remove_urlpath('test_ones.zarr')


# full
@pytest.mark.parametrize(
    "fill_value, shape, chunks, dtype, contiguous, urlpath",
    [
        (8.34, [123, 432, 222], [24, 31, 15], np.float64, True, None),
        (2.00001, [567, 375], [52, 16], np.float32, False, "test_full_sparse.iarr"),
        (8, [10, 12, 5], [5, 5, 5], np.int32, True, "test_full_contiguous.iarr"),
        (True, [12, 16], [12, 16], np.bool_, False, None),
    ],
)
def test_full_zproxy(fill_value, shape, chunks, dtype, contiguous, urlpath):
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
    ia.remove_urlpath('test_full.zarr')


@pytest.mark.parametrize(
    "shape, chunks",
    [
        ([100, 100], [50, 50]),
        ([20, 60, 30, 50], [10, 40, 10, 11]),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint16])
@pytest.mark.parametrize(
    "contiguous, urlpath, urlpath2",
    [
        (False, None, None),
        (False, None, None),
        (True, None, None),
        (True, "test_copy.iarr", "test_copy2.iarr"),
        (False, "test_copy.iarr", "test_copy2.iarr"),
    ],
)
def test_copy_zproxy(shape, chunks, dtype, contiguous, urlpath, urlpath2):
    ia.remove_urlpath(urlpath)
    ia.remove_urlpath(urlpath2)

    z = zarr.open('test_copy.zarr', mode='w', shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.arange(start=0, stop=np.prod(shape), dtype=z.dtype).reshape(shape)

    a_ = ia.zarr_proxy('test_copy.zarr', chunks=chunks, contiguous=contiguous, urlpath=urlpath)
    sl = tuple([slice(0, s - 1) for s in shape])
    a = a_[sl]
    b = a.copy(urlpath=urlpath2)
    an = ia.iarray2numpy(a)
    bn = ia.iarray2numpy(b)

    if dtype in [np.float64, np.float32]:
        rtol = 1e-6 if dtype == np.float32 else 1e-14
        np.testing.assert_allclose(an, bn, rtol=rtol)
    else:
        np.testing.assert_array_equal(an, bn)

    c = a.copy(favor=ia.Favor.SPEED)
    assert c.cfg.btune is True
    with pytest.raises(ValueError):
        a.copy(favor=ia.Favor.CRATIO, btune=False)
    d = a.copy()
    assert d.cfg.btune is False

    with pytest.raises(IOError):
        a.copy(mode="r")
    with pytest.raises(IOError):
        a.copy(mode="r+")

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath(urlpath2)
    ia.remove_urlpath('test_copy.zarr')


# cloud array
@pytest.mark.parametrize(
    "zarr_path, contiguous, urlpath",
    [
        (
            "s3://era5-pds/zarr/1987/10/data/" +
            "precipitation_amount_1hour_Accumulation.zarr/precipitation_amount_1hour_Accumulation",
            False,
            "test_cloud_proxy.iarr",
        ),
    ],
)
def test_cloud_zproxy(zarr_path, contiguous, urlpath):
    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy(zarr_path, contiguous=contiguous, urlpath=urlpath)
    b = ia.iarray2numpy(a[:5, :5, :5])
    s3 = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=zarr_path, s3=s3, check=False)
    z = zarr.open(store)
    z1 = z[:5, :5, :5]
    np.testing.assert_almost_equal(b, z1)
    ia.remove_urlpath(urlpath)
