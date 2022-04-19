import pytest
import numpy as np
import iarray as ia
from iarray import udf


@udf.scalar(verbose=0)
def fsum(a: udf.float64, b: udf.float64) -> float:
    return a + b


@udf.scalar(verbose=0)
def fmult(a: udf.float64, b: udf.float64) -> float:
    return a * b


@pytest.mark.parametrize(
    "sexpr, sexpr_udf, inputs",
    [
        ("x + x", "lib.fsum(x, x)", {"x": ia.arange((10,))}),
        ("x * x", "lib.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 * (x + x)", "2 * lib.fsum(x, x)", {"x": ia.arange((10,))}),
        ("2 + x * x", "2 + lib.fmult(x, x)", {"x": ia.arange((10,))}),
        # ("2 + x * x * (x + x)", "2 + lib.fmult(x, x) * lib.fsum(x, x)", {"x": ia.arange((10,))}),  # segfaults!
        # ("2 + x * x", "2 + lib.fmult2(x, x)", {"x": ia.arange((10,))}),  # segfaults (should be function not found!)
        # ("2 + x * x", "2 + lib2.fmult(x, x)", {"x": ia.arange((10,))}),  # segfaults (should be lib2 not found!)
    ],
)
def test_simple(sexpr, sexpr_udf, inputs):
    libs = ia.UdfLibraries()
    libs["lib"].register_func(fsum)
    libs["lib"].register_func(fmult)
    expr = ia.expr_from_string(sexpr, inputs)
    expr_udf = ia.expr_from_string(sexpr_udf, inputs)
    out = expr.eval()
    out_udf = expr_udf.eval()
    tol = 1e-14
    np.testing.assert_allclose(out.data, out_udf.data, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "sexpr_udf, inputs",
    [
        ("lib.fsum(x1, x)", {"x": ia.arange((10,))}),
        ("lib.fsum(x1, x1)", {"x": ia.arange((10,))}),
    ],
)
def test_malformed(sexpr_udf, inputs):
    libs = ia.UdfLibraries()
    libs["lib"].register_func(fsum)
    libs["lib"].register_func(fmult)
    with pytest.raises(ValueError):
        expr_udf = ia.expr_from_string(sexpr_udf, inputs)
