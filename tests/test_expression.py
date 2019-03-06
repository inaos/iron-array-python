import pytest
import iarray as ia
import numpy as np
import numexpr as ne


@pytest.mark.parametrize("eval_flags, shape, pshape, dtype, expression",
                         [
                             ("block", [10*1000], [1000], "double", "(x - 1.35) * (x - 4.45) * (x - 8.5)"),
                             ("iterchunk", [100*100], [100], "float", "(x - 1) * (x - 1) + 2 * x"),
                         ])
def test_expression(eval_flags, shape, pshape, dtype, expression):
    cfg = ia.Config(eval_flags=eval_flags)
    ctx = ia.Context(cfg)

    size = int(np.prod(shape))
    a = ia.linspace(ctx, size, 0, 10, shape, pshape, dtype)
    b = ia.iarray2numpy(ctx, a)

    e = ia.Expression(ctx)
    e.bind("x".encode("utf-8"), a)
    e.compile(expression.encode("utf-8"))
    c = e.eval(shape, pshape, dtype)
    d = ia.iarray2numpy(ctx, c)

    f = ne.evaluate(expression, local_dict={"x": b})

    np.testing.assert_almost_equal(d, f)
