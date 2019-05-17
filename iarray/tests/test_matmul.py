import pytest
import iarray as ia
import numpy as np
import numexpr as ne


# Expression
@pytest.mark.parametrize("ashape, apshape, abshape, bshape, bpshape, bbshape, dtype",
                         [
                             ([100, 100], [20, 20], [23, 32], [100, 100], [30, 30], [32, 23], "double"),
                             ([100, 100], None, [100, 100], [100, 100], None, [100, 100], "float"),
                             ([100, 100],  [40, 40], [23, 32], [100], [12], [32], "double"),
                             ([100, 100], None, [100, 100], [100], None, [100], "float"),
                             ([100, 100], None, [100, 100], [100, 100], [20, 20], [100, 23], "double"),
                             ([100, 100], [20, 20], [80, 100], [100, 100], None, [100, 100], "float"),
                             ([100, 100], None, [100, 100], [100], [5], [100], "double"),
                             ([100, 100], [30, 30], [12, 100], [100], None, [100], "float")
                         ])
def test_matmul(ashape, apshape, abshape, bshape, bpshape, bbshape, dtype):
    cfg = ia.Config()
    ctx = ia.Context(cfg)

    asize = int(np.prod(ashape))
    a = ia.linspace(ctx, asize, -10, 10, ashape, apshape, dtype)
    an = ia.iarray2numpy(ctx, a)

    bsize = int(np.prod(bshape))
    b = ia.linspace(ctx, bsize, -10, 10, bshape, bpshape, dtype)
    bn = ia.iarray2numpy(ctx, b)

    c = ia.matmul(ctx, a, b, abshape, bbshape)
    cn = np.matmul(an, bn)

    cn_2 = ia.iarray2numpy(ctx, c)

    np.testing.assert_almost_equal(cn, cn_2)
