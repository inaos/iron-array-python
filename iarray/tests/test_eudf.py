import numpy as np
import pytest

import iarray as ia
from iarray.eudf import eudf


@pytest.mark.parametrize(
    "sexpr, inputs",
    [
        ("x + x", {"x": ia.arange((10,))}),
        ("x * x", {"x": ia.arange((10,))}),
#       ("x + y", {"x": ia.arange((10,)), "y": 1}),  # scalar as param!
        ("2 * (x + x)", {"x": ia.arange((10,))}),
        ("2 + x * x", {"x": ia.arange((10,))}),
#       ("2 + sin(x) + x * x", {"x": ia.arange((10,))}),
#       ("2 * (sin(x) + cos(x)) + x * x", {"x": ia.arange((10,))}),
        ("2 + x * x * (x + x)", {"x": ia.arange((10,))}),
        ("2 + x * x * ((x * x) + x)", {"x": ia.arange((10,))}),
#       ("x * y * ((x * y) + y)", {"x": ia.arange((10,)), "y": 2}),
    ],
)
def test_simple(sexpr, inputs):
    expr = ia.expr_from_string(sexpr, inputs)
    out = expr.eval()
    expr_udf = eudf(sexpr, inputs)
    out_udf = expr_udf.eval()
    tol = 1e-14
    np.testing.assert_allclose(out.data, out_udf.data, rtol=tol, atol=tol)
