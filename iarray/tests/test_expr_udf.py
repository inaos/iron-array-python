import numpy as np
import pytest

import iarray as ia
from iarray import udf
from iarray.expr_udf import expr_udf


TOL = 1e-14

@udf.scalar(lib="lib_expr_udf")
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
        ("lib_expr_udf.fsum(x, x)", {"x": ia.arange((10,))}),
        ("x + y", {"x": ia.arange((10, 10)), "y": ia.arange((10, 10))}),
    ],
)
def test_simple(sexpr, inputs):
    out = expr_udf(sexpr, inputs).eval()
    ref_out = ia.expr_from_string(sexpr, inputs).eval()
    np.testing.assert_allclose(out.data, ref_out.data, rtol=TOL, atol=TOL)


@pytest.mark.parametrize(
    "sexpr, sexpr_np, inputs",
    [
        ('a[b > 5.0]', 'b > 5.0', {'a': ia.arange([10]), 'b': ia.arange([10])}),
    ]
)
def test_complex(sexpr, sexpr_np, inputs):
    out = expr_udf(sexpr, inputs).eval()

    a = inputs['a'].data
    b = inputs['b'].data
    out_ref = np.where(eval(sexpr_np), a, np.nan)

    np.testing.assert_allclose(out.data, out_ref, rtol=TOL, atol=TOL)
