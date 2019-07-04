import pytest
import iarray as ia
import numpy as np


# Expression
@pytest.mark.parametrize("eval_flags, shape, pshape, dtype, expression",
                         [
                             ("iterblock", [1000], [100], np.float64, "(x - 1.35) * (x - 4.45) * (x - 8.5)"),
                             ("iterchunk", [1000], [123], np.float32, "(x - 1) * (x - 1) + 2 * x"),
                             ("iterblock", [100, 100], [23, 32], np.float64, "(x - 1) * 2"),
                             ("iterchunk", [100, 100, 55], [10, 5, 10], np.float32, "(x - 1) * (x - 1) + 2 * x"),
                             ("iterblock", [1000], None, np.float64, "(x - 1.35) * (x - 4.45) * (x - 8.5)"),
                             ("iterchunk", [1000], None, np.float32, "(x - 1) * (x - 1) + 2 * x"),
                             ("iterblock", [100, 100], None, np.float64, "(x - 1) * 2"),
                             ("iterchunk", [8, 6, 7, 4, 5], None, np.float32, "(x - 1) * (x - 1) + 2 * x")
                         ])
def test_expression(eval_flags, shape, pshape, dtype, expression):
    a = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 10)
    b = ia.iarray2numpy(a)

    e = ia.Expr(eval_flags=eval_flags)
    e.bind("x", a)
    e.compile(expression)
    c = e.eval(shape, pshape, dtype)
    d = ia.iarray2numpy(c)

    f = eval(expression, {"x": b})

    np.testing.assert_almost_equal(d, f)
