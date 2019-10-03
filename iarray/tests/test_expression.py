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

    expr = ia.Expr(eval_flags=eval_flags)
    expr.bind("x", x)
    expr.bind("y", y)
    expr.compile(expression)
    iout = expr.eval(shape, pshape, dtype)
    npout = ia.iarray2numpy(iout)

    # npout2 = eval(expression, {"x": npx})  # pure numpy
    # npout2 = ne.evaluate(expression, {"x": npx})  # numexpr
    npout2 = parser.parse(expression).evaluate({"x": npx, "y": npy})
    np.testing.assert_almost_equal(npout, npout2)


    # IARRAY_FUNC_ABS,
    # IARRAY_FUNC_ACOS,
    # IARRAY_FUNC_ASIN,
    # IARRAY_FUNC_ATAN,
    # IARRAY_FUNC_ATAN2,
    # IARRAY_FUNC_CEIL,
    # IARRAY_FUNC_COS,
    # IARRAY_FUNC_COSH,
    # IARRAY_FUNC_E,
    # IARRAY_FUNC_EXP,
    # IARRAY_FUNC_FAC,
    # IARRAY_FUNC_FLOOR,
    # IARRAY_FUNC_LN,
    # IARRAY_FUNC_LOG,
    # IARRAY_FUNC_LOG10,
    # IARRAY_FUNC_NCR,
    # IARRAY_FUNC_NEGATE,
    # IARRAY_FUNC_NPR,
    # IARRAY_FUNC_PI,
    # IARRAY_FUNC_POW,
    # IARRAY_FUNC_SIN,
    # IARRAY_FUNC_SINH,
    # IARRAY_FUNC_SQRT,
    # IARRAY_FUNC_TAN,
    # IARRAY_FUNC_TANH,

# ufuncs
@pytest.mark.parametrize("ufunc, ia_expr, dtype",
                         [
                             # ("np.abs(x)", "abs(x)", np.float64),  # TODO: TypeError: bad operand type for abs(): 'IArray'
                             ("np.arccos(x)", "acos(x)", np.float64),
                             ("np.arcsin(x)", "asin(x)", np.float64),
                             ("np.arctan(x)", "atan(x)", np.float64),
                             ("np.arctan2(x, y)", "atan2(x, y)", np.float64),
                             # ("np.ceil(x)", "ceil(x)", np.float32), # TODO: TypeError: must be real number, not IArray
                             ("np.cos(x)", "cos(x)", np.float64),
                             ("np.cosh(x)", "cosh(x)", np.float64),
                             ("np.exp(x)", "exp(x)", np.float64),
                             # ("np.floor(x)", "floor(x)", np.float64), # TODO: TypeError: must be real number, not IArray
                             ("np.log(x)", "log(x)", np.float64),
                             ("np.log10(x)", "log10(x)", np.float64),
                             # ("np.negative(x)", "negate(x)", np.float64),  # TODO: TypeError: bad operand type for unary -: 'IArray'
                             # ("np.power(x, y)", "pow(x, y)", np.float64),  # TODO: TypeError: unsupported operand type(s) for ** or pow(): 'IArray' and 'IArray'
                             ("np.sin(x)", "sin(x)", np.float32),
                             ("np.sinh(x)", "sinh(x)", np.float64),
                             ("np.sqrt(x)", "sqrt(x)", np.float64),
                             ("np.tan(x)", "tan(x)", np.float64),
                             ("np.tanh(x)", "tanh(x)", np.float64),
                         ])
def test_ufuncs(ufunc, ia_expr, dtype):
    shape = [20, 30]
    pshape = [2, 3]
    # The ranges below are important for not overflowing operations
    x = ia.linspace(ia.dtshape(shape, pshape, dtype), .1, .4)
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
    np.testing.assert_almost_equal(npout, npout2)

    npout3 = eval(ufunc, {"np": np, "x": npx, "y": npy})  # pure numpy
    np.testing.assert_almost_equal(npout2, npout3)
