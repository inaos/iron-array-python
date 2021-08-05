import pytest
import iarray as ia
import numpy as np


# Matmul
@pytest.mark.parametrize(
    "use_mkl",
    [True, False],
)
@pytest.mark.parametrize(
    "ashape, achunks, ablocks," "bshape, bchunks, bblocks," "cchunks, cblocks, dtype",
    [
        ([200, 256], [100, 64], [20, 16], [256], [64], [16], [100], [20], np.float64),
        ([1024, 1024], [512, 256], [32, 32], [1024], [256], [32], [512], [32], np.float32),
    ],
)
def test_matmul(
    ashape,
    achunks,
    ablocks,
    bshape,
    bchunks,
    bblocks,
    cchunks,
    cblocks,
    dtype,
    use_mkl,
):
    astore = ia.Store(achunks, ablocks)
    a = ia.linspace(ashape, -10, 1, dtype=dtype, store=astore)
    an = ia.iarray2numpy(a)

    bstore = ia.Store(bchunks, bblocks)
    b = ia.linspace(bshape, -1, 10, dtype=dtype, store=bstore)
    bn = ia.iarray2numpy(b)

    cstore = ia.Store(chunks=cchunks, blocks=cblocks)
    c = ia.opt_gemv(a, b, store=cstore)
    cn_2 = ia.iarray2numpy(c)

    cn = np.matmul(an, bn)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(cn, cn_2, rtol=rtol)
