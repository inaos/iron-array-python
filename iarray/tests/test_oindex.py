import pytest
import iarray as ia
import numpy as np


# Slice

slice_data = [
    ([30, 100], [20, 20], [10, 13], True, None),
    ([30, 130], [50, 50], [20, 25], False, None),
    pytest.param(
        [20, 78, 55, 21],
        [3, 30, 30, 21],
        [3, 12, 6, 21],
        True,
        "test_slice_acontiguous.iarr",
        marks=pytest.mark.heavy,
    ),
    ([30, 100], [30, 44], [30, 2], False, "test_slice_asparse.iarr"),
]


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int64, np.int32, np.uint64, np.uint32]
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, acontiguous, aurlpath",
    slice_data,
)
def test_oindex(shape, chunks, blocks, dtype, acontiguous, aurlpath):
    ia.remove_urlpath(aurlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)

    a = ia.full(shape, 3.14, cfg=cfg, mode="w", dtype=dtype)
    a_copy = a.copy()

    selection = [np.random.choice(np.arange(s), 40) for s in shape]

    a.set_orthogonal_selection(selection, 0)
    b = a.get_orthogonal_selection(selection)
    c = a.oindex[selection]
    np.testing.assert_almost_equal(b, 0)
    np.testing.assert_almost_equal(c, 0)

    a_copy.oindex[selection] = 0
    b = a_copy.get_orthogonal_selection(selection)
    c = a_copy.oindex[selection]
    np.testing.assert_almost_equal(b, 0)
    np.testing.assert_almost_equal(c, 0)

    if dtype == np.float64 and a.ndim == 2:
        a_copy.oindex[[0, 2], [1, 3]] = [[-1., -2.], [-3., -4.]]
        np.testing.assert_almost_equal(a_copy.oindex[[0, 2], [1, 3]], [[-1., -2.], [-3., -4.]])
