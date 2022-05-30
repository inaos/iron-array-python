import iarray as ia
import pytest
import numpy as np
from msgpack import packb


@pytest.mark.parametrize(
    "contiguous",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, urlpath, dtype",
    [
        ([556], [221], [33], "test_attr00.iarr", np.float64),
        ([20, 134, 13], [12, 66, 8], [3, 13, 5], "test_attr01.iarr", np.int16),
        ([12, 13, 14, 15, 16], [8, 9, 4, 12, 9], [2, 6, 4, 5, 4], None, np.float32),
    ],
)
def test_attrs(shape, chunks, blocks, urlpath, contiguous, dtype):
    ia.remove_urlpath(urlpath)

    numpy_attr = {b"dtype": str(np.dtype(dtype))}
    test_attr = {b"lorem": 1234}
    byte_attr = b"1234"

    a = ia.empty(
        shape, dtype=dtype, chunks=chunks, blocks=blocks, urlpath=urlpath, contiguous=contiguous
    )
    # setitem, len
    assert a.attrs.__len__() == 0
    a.attrs["numpy"] = numpy_attr
    assert a.attrs.__len__() == 1
    a.attrs["test"] = test_attr
    assert a.attrs.__len__() == 2

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
    keys = ["numpy", "test"]
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
    vals = [numpy_attr, test_attr]
    i = 0
    for val in a.attrs.values():
        assert val == vals[i]
        i += 1

    # pop
    elem = a.attrs.pop("test")
    assert a.attrs.__len__() == 1
    assert elem == test_attr

    # delitem
    del a.attrs["numpy"]
    assert "numpy" not in a.attrs

    # clear
    a.attrs["numpy"] = numpy_attr
    a.attrs["test"] = test_attr
    assert a.attrs.__len__() == 2
    a.attrs.clear()
    assert a.attrs.__len__() == 0

    # setdefault
    a.attrs.setdefault("default_attr", byte_attr)
    assert a.attrs["default_attr"] == byte_attr
    a.attrs["numpy"] = numpy_attr
    a.attrs.setdefault("numpy", byte_attr)
    assert a.attrs["numpy"] == numpy_attr

    # popitem
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
