import os
import pytest
import numpy as np
import iarray as ia


# linspace
@pytest.mark.parametrize(
    "start, stop, shape, chunks, blocks, dtype",
    [
        (0, 1000, [231, 120, 500], [53, 21, 340], [32, 13, 70], np.float64),
        (-1, -2000, [40, 39, 52, 120], [40, 17, 6, 50], [13, 4, 6, 50], np.float32),
    ],
)
def test_btune(start, stop, shape, chunks, blocks, dtype):

    store = ia.Store(chunks, blocks)
    with ia.config(favor=ia.Favors.SPEED, btune=True):
        a = ia.linspace(shape, start, stop, dtype=dtype, store=store)
        c1 = a.cratio

    with ia.config(favor=ia.Favors.CRATIO, btune=True):
        a = ia.linspace(shape, start, stop, dtype=dtype, store=store)
        c2 = a.cratio

    # Sometimes, depending on the machine and its state, SPEED can get better cratios :-/
    # Hopefully the 2x factor would avoid a failure in most of the cases...
    assert c1 < 2 * c2
