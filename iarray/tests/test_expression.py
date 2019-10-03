import pytest
import iarray as ia
import numpy as np


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

    expr = ia.Expr(eval_flags=eval_flags)
    expr.bind("x", x)
    expr.bind("y", y)
    expr.compile(expression)
    iout = expr.eval(shape, pshape, dtype)
    npout = ia.iarray2numpy(iout)

    npout2 = ia.Parser().parse(expression).evaluate({"x": npx, "y": npy})
    np.testing.assert_almost_equal(npout, npout2)


# ufuncs
@pytest.mark.parametrize("ufunc, ia_expr",
                         [
                             # ("np.abs(x)", "abs(x)"),  # TODO: TypeError: bad operand type for abs(): 'IArray'
                             ("np.arccos(x)", "acos(x)"),
                             ("np.arcsin(x)", "asin(x)"),
                             ("np.arctan(x)", "atan(x)"),
                             ("np.arctan2(x, y)", "atan2(x, y)"),
                             # ("np.ceil(x)", "ceil(x)"), # TODO: TypeError: must be real number, not IArray
                             ("np.cos(x)", "cos(x)"),
                             ("np.cosh(x)", "cosh(x)"),
                             ("np.exp(x)", "exp(x)"),
                             # ("np.floor(x)", "floor(x)"), # TODO: TypeError: must be real number, not IArray
                             ("np.log(x)", "log(x)"),
                             ("np.log10(x)", "log10(x)"),
                             # ("np.negative(x)", "negate(x)"),  # TODO: TypeError: bad operand type for unary -: 'IArray'
                             # ("np.power(x, y)", "pow(x, y)"),  # TODO: TypeError: unsupported operand type(s) for ** or pow(): 'IArray' and 'IArray'
                             ("np.sin(x)", "sin(x)"),
                             ("np.sinh(x)", "sinh(x)"),
                             ("np.sqrt(x)", "sqrt(x)"),
                             ("np.tan(x)", "tan(x)"),
                             ("np.tanh(x)", "tanh(x)"),
                         ])
def test_ufuncs(ufunc, ia_expr):
    shape = [20, 30]
    pshape = [2, 3]
    for dtype in np.float64, np.float32:
        # The ranges below are important for not overflowing operations
        x = ia.linspace(ia.dtshape(shape, pshape, dtype), .1, .9)
        y = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 1)
        npx = ia.iarray2numpy(x)
        npy = ia.iarray2numpy(y)

        expr = ia.Expr()
        expr.bind("x", x)
        expr.bind("y", y)
        expr.compile(ia_expr)
        iout = expr.eval(shape, pshape, dtype)
        npout = ia.iarray2numpy(iout)

        lazy_expr = eval(ufunc, {"np": np, "x": x, "y": y})
        iout2 = lazy_expr.eval(pshape=pshape, dtype=dtype)  # pure ironarray eval
        npout2 = ia.iarray2numpy(iout2)
        decimal = 6 if dtype is np.float32 else 7
        np.testing.assert_almost_equal(npout, npout2, decimal=decimal)

        npout3 = eval(ufunc, {"np": np, "x": npx, "y": npy})  # pure numpy
        np.testing.assert_almost_equal(npout2, npout3, decimal=decimal)
