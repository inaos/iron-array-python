import pytest
import numpy as np
import iarray as ia


# linspace
@pytest.mark.parametrize(
    "start, stop, shape, chunks, blocks, dtype, contiguous, urlpath",
    [
        (0, 1000, [131, 120], [53, 21], [32, 13], np.float64, False, None),
        pytest.param(
            0,
            1000,
            [231, 120, 500],
            [53, 21, 340],
            [32, 13, 70],
            np.float64,
            False,
            None,
            marks=pytest.mark.heavy,
        ),
        pytest.param(
            0,
            1000,
            [231, 120, 500],
            [53, 21, 340],
            [32, 13, 70],
            np.float64,
            True,
            "test_btune_64.iarr",
            marks=pytest.mark.heavy,
        ),
        pytest.param(
            -1,
            -2000,
            [40, 39, 52, 120],
            [40, 17, 6, 50],
            [13, 4, 6, 50],
            np.float32,
            False,
            "test_btune_32.iarr",
            marks=pytest.mark.heavy,
        ),
        (
            -1,
            -2000,
            [
                40,
                39,
            ],
            [40, 17],
            [13, 4],
            np.float32,
            True,
            None,
        ),
        pytest.param(
            -1,
            -2000,
            [40, 39, 52, 120],
            [40, 17, 6, 50],
            [13, 4, 6, 50],
            np.float32,
            True,
            None,
            marks=pytest.mark.heavy,
        ),
    ],
)
def test_btune(start, stop, shape, chunks, blocks, dtype, contiguous, urlpath):

    ia.remove_urlpath(urlpath)
    with ia.config(favor=ia.Favor.SPEED, btune=True):
        a = ia.linspace(
            start,
            stop,
            np.prod(shape),
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            blocks=blocks,
            contiguous=contiguous,
            urlpath=urlpath,
        )
        c1 = a.cratio

    ia.remove_urlpath(urlpath)
    with ia.config(favor=ia.Favor.CRATIO, btune=True):
        a = ia.linspace(
            start,
            stop,
            np.prod(shape),
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            blocks=blocks,
            contiguous=contiguous,
            urlpath=urlpath,
        )
        c2 = a.cratio

    # Sometimes, depending on the machine and its state, SPEED can get better cratios :-/
    # Hopefully the 2x factor would avoid a failure in most of the cases...
    assert c1 < 2 * c2

    ia.remove_urlpath(urlpath)
