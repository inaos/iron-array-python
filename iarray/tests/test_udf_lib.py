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
        ("x * x", "lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 * (x + x)", "2 * lib.fsum(x, x)", {"x": ia.arange((10,))}),
        ("2 + x * x", "2 + lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 + sin(x) + x * x", "2 + sin(x) + lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 * (sin(x) + cos(x)) + x * x", "2 * (sin(x) + cos(x)) + lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        # ("2 + x * x * (x + x)", "2 + lib2.fmult(x, x) * lib.fsum(x, x)", {"x": ia.arange((10,))}), # raises an error
    ],
)
def test_simple(sexpr, sexpr_udf, inputs):
    ia.udf_registry["lib"] = fsum
    assert "lib" in ia.udf_registry
    assert {"fsum"} == set(f.name for f in ia.udf_registry["lib"])
    assert "fsum" in (f.name for f in ia.udf_registry.iter_funcs("lib"))
    assert "lib.fsum" in (f for f in ia.udf_registry.iter_all_func_names())
    assert ia.udf_registry.get_func_addr("lib.fsum") != 0  # this can be any integer

    ia.udf_registry["lib2"] = fmult
    assert "lib2" in ia.udf_registry
    assert {"fmult"} == set(f.name for f in ia.udf_registry["lib2"])
    assert "fmult" in (f.name for f in ia.udf_registry.iter_funcs("lib2"))
    assert "lib2.fmult" in (f for f in ia.udf_registry.iter_all_func_names())
    assert {"lib.fsum", "lib2.fmult"} == set(ia.udf_registry.iter_all_func_names())
    assert ia.udf_registry.get_func_addr("lib2.fmult") != 0  # this can be any integer

    expr = ia.expr_from_string(sexpr, inputs)
    expr_udf = ia.expr_from_string(sexpr_udf, inputs)
    out = expr.eval()
    out_udf = expr_udf.eval()
    tol = 1e-14
    np.testing.assert_allclose(out.data, out_udf.data, rtol=tol, atol=tol)
    # The registry for udf funcs is global, so make sure to clear it once is used
    ia.udf_registry.clear()


@pytest.mark.parametrize(
    "sexpr_udf, inputs",
    [
        ("lib.fsum(x1, x)", {"x": ia.arange((10,))}),
        ("lib.fsum(x1, x1)", {"x": ia.arange((10,))}),
        ("2 + lib.fmult2(x, x)", {"x": ia.arange((10,))}),
        ("2 + lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 + sin(x) + lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 + lib.fmult(x, x) * lib2.fsum(x, x)", {"x": ia.arange((10,))}),
    ],
)
def test_malformed(sexpr_udf, inputs):
    ia.udf_registry["lib"] = fsum
    ia.udf_registry["lib"] = fmult
    with pytest.raises(ValueError):
        expr_udf = ia.expr_from_string(sexpr_udf, inputs)
    # The registry for udf funcs is global, so make sure to clear it once is used
    ia.udf_registry.clear()
