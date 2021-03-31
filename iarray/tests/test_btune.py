import os
import pytest
import numpy as np
import iarray as ia


# linspace
@pytest.mark.parametrize(
    "start, stop, shape, chunkshape, blockshape, dtype",
    [
        (0, 10, [100, 120, 50], [33, 21, 34], [12, 13, 7], np.float64),
        (-0.1, -0.2, [40, 39, 52, 12], [12, 17, 6, 5], [5, 4, 6, 5], np.float32),
    ],
)
def test_btune(start, stop, shape, chunkshape, blockshape, dtype):

    with ia.config(favor=ia.Favors.SPEED, btune=True) as cfg:
        storage = ia.Storage(chunkshape, blockshape)
        a = ia.linspace(ia.DTShape(shape, dtype), start, stop, storage=storage, cfg=cfg)
        c1 = a.cratio

    with ia.config(favor=ia.Favors.CRATIO, btune=True) as cfg:
        storage = ia.Storage(chunkshape, blockshape)
        a = ia.linspace(ia.DTShape(shape, dtype), start, stop, storage=storage, cfg=cfg)
        c2 = a.cratio

    assert c1 < c2
