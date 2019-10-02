import pytest
import iarray as ia
import numpy as np
#import numexpr as ne

parser = ia.Parser()


# Expression
@pytest.mark.parametrize("eval_flags, shape, pshape, dtype, expression",
                         [
                             ("iterblock", [1000], [100], np.float64, "(x - 1.35) * (x - 4.45) * (x - 8.5)"),
                             ("iterblock", [1000], [100], np.float64, "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)"),
                             ("iterblosc", [1000], [100], np.float64, "(cos(x) - 1.35) * (sin(x) - 4.45) * tan(x - 8.5)"),
                             ("iterchunk", [1000], [123], np.float32, "(abs(-x) - 1.35) * ceil(x) * floor(x - 8.5)"),
                             ("iterblock", [100, 100], [23, 32], np.float64, "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)"),
                             ("iterchunk", [100, 100, 55], [10, 5, 10], np.float32, "asin(x) + (acos(x) - 1.35) - atan(x + .2)"),
                             ("iterblock", [1000], None, np.float64, "exp(x) + (log(x) - 1.35) - log10(x + .2)"),
                             ("iterchunk", [1000], None, np.float32, "sqrt(x) + atan2(x, x) + pow(x, x)"),
                             ("iterblock", [100, 100], None, np.float64, "(x - cos(1)) * 2"),
                             ("iterchunk", [8, 6, 7, 4, 5], None, np.float32, "(x - cos(y)) * (sin(x) + y) + 2 * x + y"),
                             ("iterblosc", [8, 6, 7, 4, 5], [4, 3, 3, 4, 5], np.float32, "(x - cos(y)) * (sin(x) + y) + 2 * x + y"),
                         ])
def test_expression(eval_flags, shape, pshape, dtype, expression):
    # The ranges below are important for not overflowing operations
    x = ia.linspace(ia.dtshape(shape, pshape, dtype), .001, .2)
    y = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 1)
    npx = ia.iarray2numpy(x)
    npy = ia.iarray2numpy(y)

    e = ia.Expr(eval_flags=eval_flags)
    e.bind("x", x)
    e.bind("y", y)
    e.compile(expression)
    c = e.eval(shape, pshape, dtype)
    d = ia.iarray2numpy(c)

    # f = eval(expression, {"x": npx})  # pure numpy
    # f = ne.evaluate(expression, {"x": npx})  # numexpr
    f = parser.parse(expression).evaluate({"x": npx, "y": npy})

    np.testing.assert_almost_equal(d, f)
