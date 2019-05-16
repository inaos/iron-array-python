import pytest
import iarray as ia
import numpy as np
import numexpr as ne


# Expression
@pytest.mark.parametrize("ashape, apshape, bshape, bpshape, dtype",
                         [
                             ([100, 100], [23, 32], [100, 100], [32, 23], "double"),
                             ([100, 100], None, [100, 100], None, "float"),
                             ([100, 100], [23, 32], [100], [32], "double"),
                             ([100, 100], None, [100], None, "float")
                         ])
def test_matmul(ashape, apshape, bshape, bpshape, dtype):
    cfg = ia.Config()
    ctx = ia.Context(cfg)

    asize = int(np.prod(ashape))
    a = ia.linspace(ctx, asize, 0, 10, ashape, apshape, dtype)
    an = ia.iarray2numpy(ctx, a)

    bsize = int(np.prod(bshape))
    b = ia.linspace(ctx, bsize, 0, 10, bshape, bpshape, dtype)
    bn = ia.iarray2numpy(ctx, b)

    c = ia.matmul(ctx, a, b, apshape, bpshape)
    cn = np.matmul(an, bn)

    cn_2 = ia.iarray2numpy(ctx, c)

    np.testing.assert_almost_equal(cn, cn_2)
