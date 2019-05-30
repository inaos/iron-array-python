import pytest
import iarray as ia
import numpy as np


# Expression
@pytest.mark.parametrize("eval_flags, shape, pshape, dtype, expression",
                         [
                             ("block", [1000], [100], "double", "(x - 1.35) * (x - 4.45) * (x - 8.5)"),
                             ("iterchunk", [1000], [123], "float", "(x - 1) * (x - 1) + 2 * x"),
                             ("block", [100, 100], [23, 32], "double", "(x - 1) * 2"),
                             ("iterchunk", [100, 100, 55], [10, 5, 10], "float", "(x - 1) * (x - 1) + 2 * x"),
                             ("block", [1000], None, "double", "(x - 1.35) * (x - 4.45) * (x - 8.5)"),
                             ("iterchunk", [1000], None, "float", "(x - 1) * (x - 1) + 2 * x"),
                             ("block", [100, 100], None, "double", "(x - 1) * 2"),
                             ("iterchunk", [8, 6, 7, 4, 5], None, "float", "(x - 1) * (x - 1) + 2 * x")
                         ])
def test_expression(eval_flags, shape, pshape, dtype, expression):
    cfg = ia.Config(eval_flags=eval_flags)
    ctx = ia.Context(cfg)

    size = int(np.prod(shape))
    a = ia.linspace2(size, 0, 10, shape, pshape, dtype)
    b = ia.iarray2numpy2(a)

    e = ia.Expression(ctx)
    e.bind("x", a)
    e.compile(expression)
    c = e.eval(shape, pshape, dtype)
    d = ia.iarray2numpy2(c)

    f = eval(expression, {"x": b})

    np.testing.assert_almost_equal(d, f)
