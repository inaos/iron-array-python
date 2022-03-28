import pytest
import iarray as ia
import numpy as np


# Test load, open and save
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint32])
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    [
        ([123], [44], [20]),
        ([100, 123], [12, 21], [10, 10]),
        pytest.param([100, 100], [5, 17], [5, 5], marks=pytest.mark.heavy),
        ([10, 12, 21], [5, 4, 10], [2, 1, 5]),
    ],
)
@pytest.mark.parametrize("func", [ia.load, ia.open])
def test_load_save(shape, chunks, blocks, dtype, func, contiguous):
    urlpath = "test_load_save.iarr"

    ia.remove_urlpath(urlpath)

    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=contiguous)
    max = 1
    if dtype not in [np.float64, np.float32]:
        for i in range(len(shape)):
            max *= shape[i]
    a = ia.arange(shape, 0, max, dtype=dtype, cfg=cfg)
    an = ia.iarray2numpy(a)

    ia.save(urlpath, a, contiguous=contiguous)

    b = func(urlpath)
    bn = ia.iarray2numpy(b)

    if dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(an, bn)
    else:
        np.testing.assert_array_equal(an, bn)

    # Overwrite existing array
    ia.save(urlpath, a, contiguous=contiguous)

    b = ia.open(urlpath)
    assert(b.cfg.contiguous == contiguous)
    assert(isinstance(b.cfg.urlpath, bytes))
    assert(b.cfg.urlpath == urlpath.encode("utf-8"))
    assert(b.cfg.chunks == a.chunks)
    assert(b.cfg.blocks == a.blocks)
    assert(b.cfg.filters == a.cfg.filters)
    assert(b.cfg.fp_mantissa_bits == a.cfg.fp_mantissa_bits)
    assert(b.dtype == a.dtype)
    assert(b.cfg.mode == "a")

    c = ia.load(urlpath)
    assert(c.cfg.contiguous is False)
    assert(c.cfg.urlpath is None)
    assert(c.cfg.chunks == a.chunks)
    assert(c.cfg.blocks == a.blocks)
    assert(c.cfg.codec == a.cfg.codec)
    assert(c.cfg.filters == a.cfg.filters)
    assert(c.cfg.fp_mantissa_bits == a.cfg.fp_mantissa_bits)
    assert(c.dtype == a.dtype)
    assert(c.cfg.mode == "a")

    ia.remove_urlpath(urlpath)
