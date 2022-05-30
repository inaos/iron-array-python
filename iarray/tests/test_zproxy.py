import pytest
import numpy as np
import s3fs
import zarr
from msgpack import packb
import iarray as ia


# linspace
@pytest.mark.parametrize(
    "start, stop, shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        (
            0,
            10,
            [100, 120, 50],
            [33, 21, 34],
            [10, 10, 10],
            "f8",
            False,
            "test_linspace_sparse.iarr",
        ),
        pytest.param(
            -0.1,
            -0.2,
            [40, 39, 52, 12],
            [12, 17, 6, 5],
            [5, 5, 3, 3],
            "float32",
            True,
            None,
            marks=pytest.mark.heavy,
        ),
        (
            0,
            10,
            [55, 24, 31],
            [55, 24, 15],
            [25, 24, 10],
            "float64",
            True,
            "test_linspace_contiguous.iarr",
        ),
        (-0.1, -0.2, [4, 3, 5, 2], [4, 3, 5, 2], [4, 3, 5, 2], "f4", False, None),
    ],
)
def test_linspace_zproxy(start, stop, shape, chunks, blocks, dtype, contiguous, urlpath):
    size = np.prod(shape)
    z = zarr.open("test_linspace.zarr", mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.linspace(start, stop, size, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy("test_linspace.zarr", contiguous=contiguous, urlpath=urlpath, blocks=blocks)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if z.dtype == "float64" else np.float32
    c = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_linspace.zarr")


# arange
@pytest.mark.parametrize(
    "start, stop, shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        (
            0,
            10,
            [22, 21, 51],
            [12, 14, 22],
            [12, 14, 22],
            "float64",
            False,
            "test_arange_sparse.iarr",
        ),
        pytest.param(
            0,
            1,
            [12, 12, 15, 13, 18, 19],
            [6, 5, 4, 7, 7, 5],
            [3, 3, 2, 7, 7, 5],
            "float32",
            True,
            None,
            marks=pytest.mark.heavy,
        ),
        (
            0,
            10 * 12 * 5,
            [10, 12, 5],
            [5, 5, 5],
            [5, 2, 5],
            "uint64",
            True,
            "test_arange_contiguous.iarr",
        ),
        (-0.1, -0.2, [4, 3, 5, 2], [2, 2, 2, 2], [2, 2, 2, 2], "i4", False, None),
    ],
)
def test_arange_zproxy(start, stop, shape, chunks, blocks, dtype, contiguous, urlpath):
    size = int(np.prod(shape))
    step = (stop - start) / size
    z = zarr.open("test_arange.zarr", mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.arange(start, stop, step, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy("test_arange.zarr", contiguous=contiguous, urlpath=urlpath, blocks=blocks)
    b = ia.iarray2numpy(a)
    npdtype = b.dtype
    c = np.arange(start, stop, step, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)
    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_arange.zarr")


# from_file
@pytest.mark.parametrize(
    "start, stop, shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        (0, 10, [1234], [123], [50], "float64", False, "test.fromfile0.iarr"),
        (0, 10, [12, 12], [6, 5], [6, 5], "float32", True, "test.fromfile1.iarr"),
        pytest.param(
            -0.1,
            -0.10,
            [10, 12, 21, 31, 11],
            [4, 3, 5, 5, 2],
            [2, 3, 2, 2, 1],
            "float32",
            True,
            "test.fromfile2.iarr",
            marks=pytest.mark.heavy,
        ),
    ],
)
def test_from_file_zproxy(start, stop, shape, chunks, blocks, dtype, contiguous, urlpath):
    size = np.prod(shape)
    z = zarr.open("test_linspace.zarr", mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.linspace(start, stop, size, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    ia.zarr_proxy("test_linspace.zarr", contiguous=contiguous, urlpath=urlpath, blocks=blocks)

    b = ia.load(urlpath)
    c = ia.iarray2numpy(b)
    npdtype = np.float64 if z.dtype == "float64" else np.float32
    d = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(c, d)

    e = ia.open(urlpath)
    f = ia.iarray2numpy(e)
    np.testing.assert_almost_equal(f, d)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_linspace.zarr")


# get_slice
@pytest.mark.parametrize(
    "start, stop, slice, shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        (
            0,
            10,
            (slice(2, 4), slice(5, 10), slice(1, 2)),
            [21, 31, 21],
            [10, 12, 5],
            [5, 16, 2],
            "float64",
            True,
            None,
        ),
        (
            -0.1,
            -0.2,
            (slice(2, 4), slice(7, 12)),
            [55, 123],
            [12, 16],
            [6, 8],
            "float32",
            False,
            "test_slice_sparse.iarr",
        ),
        (
            0,
            10 * 12 * 5,
            (slice(2, 4), slice(5, 10), slice(1, 2)),
            [10, 12, 5],
            [5, 6, 5],
            [3, 2, 5],
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
            [120, 40],
            np.int16,
            False,
            None,
        ),
    ],
)
def test_slice_zproxy(start, stop, slice, shape, chunks, blocks, dtype, contiguous, urlpath):
    size = int(np.prod(shape))
    step = (stop - start) / size
    z = zarr.open("test_slice.zarr", mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.arange(start, stop, step, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy("test_slice.zarr", contiguous=contiguous, urlpath=urlpath, blocks=blocks)
    b = a[slice]
    c = ia.iarray2numpy(b)

    npdtype = a.dtype
    d = np.arange(start, stop, step, dtype=npdtype).reshape(shape)[slice]
    if dtype in [np.float64, np.float32]:
        np.testing.assert_allclose(c, d)
    else:
        np.testing.assert_array_equal(c, d)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_slice.zarr")


# zeros
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        pytest.param(
            [134, 1234, 238],
            [10, 25, 35],
            [5, 15, 15],
            np.float64,
            True,
            "test_zeros_contiguous.iarr",
            marks=pytest.mark.heavy,
        ),
        ([456, 431], [102, 16], [50, 16], np.float32, False, "test_zeros_sparse.iarr"),
        ([10, 12, 5], [10, 1, 1], [10, 1, 1], np.int16, False, None),
        ([12, 16], [1, 4], [1, 3], np.uint8, True, None),
        ([12, 16], [1, 16], [1, 16], "b", True, "test_zeros_contiguous.iarr"),
    ],
)
def test_zeros_zproxy(shape, chunks, blocks, dtype, contiguous, urlpath):
    z = zarr.open("test_zeros.zarr", mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.zeros(shape=shape, dtype=z.dtype)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy("test_zeros.zarr", contiguous=contiguous, urlpath=urlpath, blocks=blocks)

    b = ia.iarray2numpy(a)
    npdtype = dtype
    c = np.zeros(shape, dtype=npdtype)
    if dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_zeros.zarr")


# ones
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        pytest.param(
            [456, 12, 234],
            [55, 6, 21],
            [6, 3, 11],
            np.float64,
            False,
            None,
            marks=pytest.mark.heavy,
        ),
        ([1024, 55], [66, 22], [22, 11], np.float32, True, "test_ones_contiguous.iarr"),
        ([10, 12, 5], [5, 6, 5], [5, 6, 5], np.int8, True, None),
        ([120, 130], [45, 64], [25, 34], np.uint16, False, "test_ones_sparse.iarr"),
        ([10, 12, 5], [5, 6, 5], [5, 6, 5], np.bool_, True, None),
    ],
)
def test_ones_zproxy(shape, chunks, blocks, dtype, contiguous, urlpath):
    z = zarr.open("test_ones.zarr", mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.ones(shape=shape, dtype=z.dtype)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy("test_ones.zarr", contiguous=contiguous, urlpath=urlpath, blocks=blocks)

    b = ia.iarray2numpy(a)
    npdtype = dtype
    c = np.ones(shape, dtype=npdtype)
    if dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_ones.zarr")


# full
@pytest.mark.parametrize(
    "fill_value, shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        pytest.param(
            8.34,
            [123, 432, 222],
            [24, 31, 15],
            [24, 31, 15],
            np.float64,
            True,
            None,
            marks=pytest.mark.heavy,
        ),
        (2.00001, [567, 375], [52, 16], [52, 16], np.float32, False, "test_full_sparse.iarr"),
        (8, [10, 12, 5], [5, 5, 5], [5, 5, 5], np.int32, True, "test_full_contiguous.iarr"),
        (True, [12, 16], [12, 16], [3, 5], np.bool_, False, None),
    ],
)
def test_full_zproxy(fill_value, shape, chunks, blocks, dtype, contiguous, urlpath):
    z = zarr.open("test_full.zarr", mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.full(shape, fill_value, dtype=z.dtype)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy("test_full.zarr", contiguous=contiguous, urlpath=urlpath, blocks=blocks)

    b = ia.iarray2numpy(a)
    npdtype = dtype
    c = np.full(shape, fill_value, dtype=npdtype)
    if dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(b, c)
    else:
        np.testing.assert_array_equal(b, c)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_full.zarr")


@pytest.mark.parametrize(
    "shape, chunks, blocks",
    [
        ([100, 100], [50, 50], [50, 50]),
        pytest.param([20, 60, 30, 50], [10, 40, 10, 11], [5, 5, 5, 5], marks=pytest.mark.heavy),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint16])
@pytest.mark.parametrize(
    "contiguous, urlpath, urlpath2",
    [
        (False, None, None),
        (True, None, None),
        (True, "test_copy.iarr", "test_copy1.iarr"),
        (False, None, "test_copy2.iarr"),
    ],
)
def test_copy_zproxy(shape, chunks, blocks, dtype, contiguous, urlpath, urlpath2):
    ia.remove_urlpath(urlpath)
    ia.remove_urlpath(urlpath2)

    z = zarr.open("test_copy.zarr", mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.arange(start=0, stop=np.prod(shape), dtype=z.dtype).reshape(shape)

    a_ = ia.zarr_proxy(
        "test_copy.zarr", chunks=chunks, blocks=blocks, contiguous=contiguous, urlpath=urlpath
    )
    sl = tuple([slice(0, s) for s in shape])
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
    ia.remove_urlpath("test_copy.zarr")


# cloud array
@pytest.mark.parametrize(
    "zarr_path, contiguous, urlpath",
    [
        pytest.param(
            "s3://era5-pds/zarr/1987/10/data/"
            + "precipitation_amount_1hour_Accumulation.zarr/precipitation_amount_1hour_Accumulation",
            False,
            "test_cloud_proxy.iarr",
            marks=pytest.mark.heavy,
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


# attrs
@pytest.mark.parametrize(
    "shape, chunks, blocks, urlpath, dtype",
    [
        ([556], [221], [33], "test_attr00.iarr", np.float64),
        ([20, 134, 13], [12, 66, 8], [3, 13, 5], "test_attr01.iarr", np.int16),
        ([12, 13, 14, 15, 16], [8, 9, 4, 12, 9], [2, 6, 4, 5, 4], None, np.float32),
    ],
)
def test_zproxy_attrs(shape, chunks, blocks, urlpath, dtype):
    ia.remove_urlpath(urlpath)

    bool_attr = False
    int_attr = 189063
    str_attr = "1234"

    zarr_urlpath = "test_attrs.zarr"
    z = zarr.open(zarr_urlpath, shape=shape, chunks=chunks, dtype=dtype, mode="w")
    assert z.attrs.__len__() == 0
    z.attrs["bool"] = bool_attr
    a = ia.zarr_proxy(zarr_urlpath=zarr_urlpath, urlpath=urlpath, dtype=dtype, mode="w")

    # setitem, len
    assert a.zarr_attrs.__len__() == 1
    a.zarr_attrs["int"] = int_attr
    assert a.zarr_attrs.__len__() == 2

    # contains
    assert "bool" in a.zarr_attrs
    assert "error" not in a.zarr_attrs
    assert a.zarr_attrs["bool"] == bool_attr
    assert "int" in a.zarr_attrs
    # getitem
    assert a.zarr_attrs["int"] == int_attr

    int_attr = 3
    a.zarr_attrs["int"] = int_attr
    assert a.zarr_attrs["int"] == int_attr

    # iter
    keys = ["bool", "int"]
    i = 0
    for attr in a.zarr_attrs:
        assert attr == keys[i]
        i += 1
    # keys
    i = 0
    for attr in a.zarr_attrs.keys():
        assert attr == keys[i]
        i += 1
    # values
    vals = [bool_attr, int_attr]
    i = 0
    for val in a.zarr_attrs.values():
        assert val == vals[i]
        i += 1

    # pop
    elem = a.zarr_attrs.pop("int")
    assert a.zarr_attrs.__len__() == 1
    assert elem == int_attr

    # delitem
    del a.zarr_attrs["bool"]
    assert "bool" not in a.zarr_attrs

    # clear
    a.zarr_attrs["bool"] = bool_attr
    a.zarr_attrs["int"] = int_attr
    assert a.zarr_attrs.__len__() == 2
    a.zarr_attrs.clear()
    assert a.zarr_attrs.__len__() == 0
    # The next line is needed so that Zarr updates its attributes
    z.attrs.refresh()
    assert z.attrs.__len__() == 0
    d = {"attr1": 12, "attr2": 3}
    a.zarr_attrs.put(d)
    assert a.zarr_attrs["attr1"] == d["attr1"]
    assert a.zarr_attrs["attr1"] == d["attr1"]
    assert a.zarr_attrs.__len__() == 2
    a.zarr_attrs.clear()

    # setdefault
    a.zarr_attrs.setdefault("default_attr", str_attr)
    assert a.zarr_attrs["default_attr"] == str_attr
    a.zarr_attrs["bool"] = bool_attr
    a.zarr_attrs.setdefault("bool", str_attr)
    assert a.zarr_attrs["bool"] == bool_attr

    # popitem
    item = a.zarr_attrs.popitem()
    assert item == ("default_attr", str_attr)
    item = a.zarr_attrs.popitem()
    assert item == ("bool", bool_attr)

    # Special characters
    a.zarr_attrs["àçø"] = bool_attr
    assert a.zarr_attrs["àçø"] == bool_attr
    a.zarr_attrs["😆"] = int_attr
    assert a.zarr_attrs["😆"] == int_attr

    a.zarr_attrs["😆"] = "😆"
    assert a.zarr_attrs["😆"] == "😆"

    numpy_attr = {b"dtype": str(np.dtype(dtype))}
    test_attr = {b"lorem": 1234}
    byte_attr = b"1234"

    # setitem, len
    assert a.attrs.__len__() == 1
    a.attrs["numpy"] = numpy_attr
    assert a.attrs.__len__() == 2
    a.attrs["test"] = test_attr
    assert a.attrs.__len__() == 3

    # contains
    assert "numpy" in a.attrs
    assert "error" not in a.attrs
    assert a.attrs["numpy"] == numpy_attr
    assert "test" in a.attrs
    # getitem
    assert a.attrs["test"] == test_attr

    test_attr = packb({b"lorem": 4231})
    a.attrs["test"] = test_attr
    assert a.attrs["test"] == test_attr

    # iter
    keys = ["zproxy_urlpath", "numpy", "test"]
    i = 0
    for attr in a.attrs:
        assert attr == keys[i]
        i += 1
    # keys
    i = 0
    for attr in a.attrs.keys():
        assert attr == keys[i]
        i += 1
    # values
    vals = [zarr_urlpath, numpy_attr, test_attr]
    i = 0
    for val in a.attrs.values():
        assert val == vals[i]
        i += 1

    # pop
    elem = a.attrs.pop("test")
    assert a.attrs.__len__() == 2
    assert elem == test_attr

    # delitem
    del a.attrs["numpy"]
    assert "numpy" not in a.attrs

    # clear
    a.attrs["numpy"] = numpy_attr
    a.attrs["test"] = test_attr
    assert a.attrs.__len__() == 3
    a.attrs.clear()
    assert a.attrs.__len__() == 1

    # setdefault
    a.attrs.setdefault("default_attr", byte_attr)
    assert a.attrs["default_attr"] == byte_attr
    a.attrs["numpy"] = numpy_attr
    a.attrs.setdefault("numpy", byte_attr)
    assert a.attrs["numpy"] == numpy_attr

    # popitem
    item = a.attrs.popitem()
    assert item == ("zproxy_urlpath", zarr_urlpath)
    item = a.attrs.popitem()
    assert item == ("default_attr", byte_attr)
    item = a.attrs.popitem()
    assert item == ("numpy", numpy_attr)

    # Special characters
    a.attrs["àçø"] = numpy_attr
    assert a.attrs["àçø"] == numpy_attr
    a.attrs["😆"] = test_attr
    assert a.attrs["😆"] == test_attr

    # objects as values
    nparray = np.arange(start=0, stop=2)
    a.attrs["àçø"] = nparray.tobytes()
    assert a.attrs["àçø"] == nparray.tobytes()
    a.attrs["😆"] = "😆"
    assert a.attrs["😆"] == "😆"
    obj = {"dtype": str(np.dtype(dtype))}
    a.attrs["dict"] = obj
    assert a.attrs["dict"] == obj

    # Remove file on disk
    ia.remove_urlpath(urlpath)
    ia.remove_urlpath(zarr_urlpath)


# type and slice view
@pytest.mark.parametrize(
    "view_dtype, shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        pytest.param(
            np.int64,
            [123, 432, 222],
            [24, 31, 15],
            [24, 31, 15],
            np.float64,
            True,
            None,
            marks=pytest.mark.heavy,
        ),
        (np.float64, [567, 375], [52, 16], [52, 16], np.float32, False, "test_full_sparse.iarr"),
        (
            np.float32,
            [10, 12, 5],
            [5, 5, 5],
            [5, 5, 5],
            np.int32,
            True,
            "test_full_contiguous.iarr",
        ),
        (np.uint64, [12, 16], [12, 16], [3, 5], np.int8, False, None),
    ],
)
def test_type_slice_zproxy(view_dtype, shape, chunks, blocks, dtype, contiguous, urlpath):
    z = zarr.open("test_type_view.zarr", mode="w", shape=shape, chunks=chunks, dtype=dtype)
    nelem = np.prod(shape)
    z[:] = np.arange(start=0, stop=nelem, dtype=z.dtype).reshape(shape)

    ia.remove_urlpath(urlpath)
    a = ia.zarr_proxy("test_type_view.zarr", contiguous=contiguous, urlpath=urlpath, blocks=blocks)

    c = a.astype(view_dtype)
    sl = tuple([slice(0, s) for s in shape])
    f = c[sl]
    b = ia.iarray2numpy(f)
    d = np.arange(start=0, stop=nelem, dtype=z.dtype).reshape(shape)
    d = d.astype(view_dtype)
    e = d[sl]
    if view_dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(b, e)
    else:
        np.testing.assert_array_equal(b, e)

    ia.remove_urlpath(urlpath)
    ia.remove_urlpath("test_type_view.zarr")
