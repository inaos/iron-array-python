import pytest
import iarray as ia
import numpy as np

array_data = [
    ([30, 100], [20, 20], [10, 13], [30, 100], None, True, None),
    ([30, 130], [50, 50], [20, 25], [40, 130], [30, 130], False, None),
    pytest.param(
        [10, 78, 55, 21],
        [3, 30, 30, 21],
        [3, 12, 6, 21],
        [5, 80, 10, 21],
        [5, 78, 10, 21],
        True,
        "test_resize_acontiguous.iarr",
        marks=pytest.mark.heavy,
    ),
    ([30, 100], [30, 44], [30, 2], [30, 10], [30, 10], False, "test_resize_asparse.iarr"),
]


@pytest.mark.parametrize(
    "dtype, np_dtype",
    [
        (np.float32, None),
        (np.uint32, ">M8[M]"),
        (np.int64, "<m8[ps]"),
        (np.int32, None),
        (np.uint64, ">i8"),
        (np.uint32, "u8"),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, newshape, start, acontiguous, aurlpath",
    array_data,
)
def test_resize(shape, chunks, blocks, newshape, start, dtype, np_dtype, acontiguous, aurlpath):
    ia.remove_urlpath(aurlpath)
    cfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous)
    a = ia.full(
        shape=shape,
        fill_value=4,
        cfg=cfg,
        mode="a",
        dtype=dtype,
        np_dtype=np_dtype,
        urlpath=aurlpath,
    )
    b = ia.full(shape=newshape, fill_value=4, cfg=cfg, mode="a", dtype=dtype, np_dtype=np_dtype)

    outshape = a.resize(newshape=newshape, start=start)
    assert outshape == tuple(newshape)

    only_shrink = True
    for i in range(0, a.ndim):
        if newshape[i] > shape[i]:
            only_shrink = False

    if not only_shrink:
        for i in range(0, a.ndim):
            if newshape[i] <= shape[i]:
                continue
            slice_start = [0] * a.ndim
            stop = newshape.copy()

            slice_start[i] = start[i]
            stop[i] = newshape[i] - shape[i]
            if slice_start[i] % chunks[i] != 0:
                slice_start[i] += chunks[i] - slice_start[i] % chunks[i]
                stop[i] += chunks[i] - slice_start[i] % chunks[i]
            if slice_start[i] > newshape[i]:
                continue
            slice_ = tuple(slice(j, k) for (j, k) in (slice_start, stop))
            b[slice_] = 0

    npa = ia.iarray2numpy(a)
    npb = ia.iarray2numpy(b)

    out_dtype = dtype if np_dtype is None else np.dtype(np_dtype)
    if out_dtype in [np.float64, np.float32]:
        rtol = 1e-6 if out_dtype == np.float32 else 1e-14
        np.testing.assert_allclose(npa, npb, rtol=rtol, atol=0)
    else:
        np.testing.assert_equal(npa, npb)

    ia.remove_urlpath(aurlpath)
