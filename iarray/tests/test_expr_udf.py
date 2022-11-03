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
        ("x + x", {"x": ia.arange(10, shape=(10,))}),
        ("x * x", {"x": ia.arange(10, shape=(10,))}),
        ("x + y", {"x": ia.arange(10, shape=(10,)), "y": 1}),  # scalar as param!
        ("2 * (x + x)", {"x": ia.arange(10, shape=(10,))}),
        ("2 + x * x", {"x": ia.arange(10, shape=(10,))}),
        ("2 + sin(x) + x * x", {"x": ia.arange(10, shape=(10,))}),
        ("2 * (sin(x) + cos(x)) + x * x", {"x": ia.arange(10, shape=(10,))}),
        ("2 + x * x * (x + x)", {"x": ia.arange(10, shape=(10,))}),
        ("2 + x * x * ((x * x) + x)", {"x": ia.arange(10, shape=(10,))}),
        ("x * y * ((x * y) + y)", {"x": ia.arange(10, shape=(10,)), "y": 2}),
        ("lib_expr_udf.fsum(x, x)", {"x": ia.arange(10, shape=(10,))}),
        ("x + y", {"x": ia.arange(100, shape=(10, 10)), "y": ia.arange(100, shape=(10, 10))}),
        ("absolute(x) + abs(x) + negative(x) + negate(x)", {"x": ia.arange(5, step=0.5, shape=[10])}),
        (
            "absolute(x) + abs(x) + negative(x) + negate(x)",
            {"x": ia.arange(10, shape=[10], dtype=np.float32)},
        ),
        (
            "arccos(x) + arcsin(x) + arctan(x) + arctan2(x, x) + power(x, x)",
            {"x": ia.arange(10, shape=[10])},
        ),
    ],
)
def test_simple(sexpr, inputs):
    out = expr_udf(sexpr, inputs).eval()
    ref_out = ia.expr_from_string(sexpr, inputs).eval()
    np.testing.assert_allclose(out.data, ref_out.data, rtol=TOL, atol=TOL)


@pytest.mark.parametrize(
    "condition, inputs",
    [
        ("b > 5", {"a": ia.arange(10, shape=[10]), "b": ia.arange(10, shape=[10])}),
        (
            "b > 5",
            {
                "a": ia.arange(10, shape=[10], dtype=np.float32),
                "b": ia.arange(10, shape=[10], dtype=np.float32),
            },
        ),
        (
            "(b > 5) and not (a > 7) or (b > 42)",
            {"a": ia.arange(100, shape=[10, 10]), "b": ia.arange(100, shape=[10, 10])},
        ),
        ("not (a == 4)", {"a": ia.arange(10, shape=[10])}),
    ],
)
def test_masks(condition, inputs):
    sexpr = f"a[{condition}]"
    out = expr_udf(sexpr, inputs).eval()

    # Numpy
    replace = {"and": "&", "or": "|", "not": "~"}
    for k, v in replace.items():
        condition = condition.replace(k, v)
    a = inputs["a"].data
    b = inputs["b"].data if "b" in inputs else None
    out_ref = np.where(eval(condition), a, np.nan)

    np.testing.assert_allclose(out.data, out_ref, rtol=TOL, atol=TOL)
