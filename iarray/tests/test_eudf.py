import numpy as np
import pytest

import iarray as ia
from iarray import udf
from iarray.eudf import eudf


@udf.scalar(lib="lib_eudf")
def fsum(a: udf.float64, b: udf.float64) -> float:
    return a + b


@pytest.mark.parametrize(
    "sexpr, inputs",
    [
        ("x + x", {"x": ia.arange((10,))}),
        ("x * x", {"x": ia.arange((10,))}),
        ("x + y", {"x": ia.arange((10,)), "y": 1}),  # scalar as param!
        ("2 * (x + x)", {"x": ia.arange((10,))}),
        ("2 + x * x", {"x": ia.arange((10,))}),
        ("2 + sin(x) + x * x", {"x": ia.arange((10,))}),
        ("2 * (sin(x) + cos(x)) + x * x", {"x": ia.arange((10,))}),
        ("2 + x * x * (x + x)", {"x": ia.arange((10,))}),
        ("2 + x * x * ((x * x) + x)", {"x": ia.arange((10,))}),
        ("x * y * ((x * y) + y)", {"x": ia.arange((10,)), "y": 2}),
        ("lib_eudf.fsum(x, x)", {"x": ia.arange((10,))}),
    ],
)
def test_simple(sexpr, inputs):
    expr = ia.expr_from_string(sexpr, inputs)
    out = expr.eval()
    expr_udf = eudf(sexpr, inputs)
    out_udf = expr_udf.eval()
    tol = 1e-14
    np.testing.assert_allclose(out.data, out_udf.data, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "sexpr, sexpr_np, inputs",
    [
        ('a[b > 5.0]', 'b > 5.0', {'a': ia.arange([10]), 'b': ia.arange([10])}),
    ]
)
def test_complex(sexpr, sexpr_np, inputs):
    expr_udf = eudf(sexpr, inputs)
    out_udf = expr_udf.eval()

    a = inputs['a'].data
    b = inputs['b'].data
    np_out = np.where(eval(sexpr_np), a, np.nan)

    tol = 1e-14
    np.testing.assert_allclose(np_out, out_udf.data, rtol=tol, atol=tol)
