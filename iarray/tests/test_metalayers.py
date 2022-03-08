import iarray as ia
import pytest
import numpy as np
from msgpack import packb


@pytest.mark.parametrize("contiguous",
                         [
                             True,
                             False,
                         ])
@pytest.mark.parametrize("shape, chunks, blocks, urlpath, dtype",
                         [
                             ([556], [221], [33], "testmeta00.iarr", np.float64),
                             ([20, 134, 13], [12, 66, 8], [3, 13, 5], "testmeta01.iarr", np.int16),
                             ([12, 13, 14, 15, 16], [8, 9, 4, 12, 9], [2, 6, 4, 5, 4], "testmeta02.iarr", np.float32)
                         ])
def test_metalayers(shape, chunks, blocks, urlpath, contiguous, dtype):
    ia.remove_urlpath(urlpath)

    numpy_meta = packb({b"dtype": str(np.dtype(dtype))})
    test_meta = packb({b"lorem": 1234})

    a = ia.empty(shape, dtype=dtype, chunks=chunks, blocks=blocks,
                  urlpath=urlpath, contiguous=contiguous)
    a.attrs["numpy"] = numpy_meta
    a.attrs["test"] = test_meta

    assert ("numpy" in a.attrs)
    assert ("error" not in a.attrs)
    assert (a.attrs["numpy"] == numpy_meta)
    assert ("test" in a.attrs)
    assert (a.attrs["test"] == test_meta)

    test_meta = packb({b"lorem": 4231})
    a.attrs["test"] = test_meta
    assert (a.attrs["test"] == test_meta)

    del a.attrs["test"]
    assert ("test" not in a.attrs)

    # Remove file on disk
    ia.remove_urlpath(urlpath)
