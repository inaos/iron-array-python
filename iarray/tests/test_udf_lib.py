import pytest
import numpy as np
import iarray as ia
from iarray import udf


@udf.scalar(lib="lib")
def fsum(a: udf.float64, b: udf.float64) -> float:
    return a + b


@udf.scalar(lib="lib2")
def fmult(a: udf.float64, b: udf.float64) -> float:
    return a * b


@pytest.mark.parametrize(
    "sexpr, sexpr_udf, inputs",
    [
        ("x + x", "lib.fsum(x, x)", {"x": ia.arange((10,))}),
        ("x * x", "lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        ("x + y", "lib.fsum(x, y)", {"x": ia.arange((10,)), "y": 1}),  # scalar as param!
        ("2 * (x + x)", "2 * lib.fsum(x, x)", {"x": ia.arange((10,))}),
        ("2 + x * x", "2 + lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 + sin(x) + x * x", "2 + sin(x) + lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 * (sin(x) + cos(x)) + x * x", "2 * (sin(x) + cos(x)) + lib2.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 + x * x * (x + x)", "2 + lib2.fmult(x, x) * lib.fsum(x, x)", {"x": ia.arange((10,))}),
        ("2 + x * x * ((x * x) + x)", "2 + lib2.fmult(x, x) * lib.fsum(lib2.fmult(x, x), x)", {"x": ia.arange((10,))}),
        ("x * y * ((x * y) + y)", "lib2.fmult(x, y) * lib.fsum(lib2.fmult(x, y), y)", {"x": ia.arange((10,)), "y": 2}),
    ],
)
def test_simple(sexpr, sexpr_udf, inputs):
    assert "lib" in ia.udf_registry
    assert {"fsum"} == set(fname for fname in ia.udf_registry["lib"])
    assert "fsum" in (f.name for f in ia.udf_registry.iter_funcs("lib"))
    assert "lib.fsum" in (f for f in ia.udf_registry.iter_all_func_names())
    assert ia.udf_registry.get_func_addr("lib.fsum") != 0  # this can be any integer

    assert "lib2" in ia.udf_registry
    assert {"fmult"} == set(fname for fname in ia.udf_registry["lib2"])
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


@pytest.mark.parametrize(
    "sexpr_udf, inputs",
    [
        ("lib.fsum(x1, x)", {"x": ia.arange((10,))}),
        ("lib.fsum(x1, x1)", {"x": ia.arange((10,))}),
        ("2 + lib.fmult2(x, x)", {"x": ia.arange((10,))}),
        ("2 + lib.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 + sin(x) + lib.fmult(x, x)", {"x": ia.arange((10,))}),
        ("2 + lib.fmult(x, x) * lib2.fsum(x, x)", {"x": ia.arange((10,))}),
    ],
)
def test_malformed(sexpr_udf, inputs):
    with pytest.raises(ValueError):
        ia.expr_from_string(sexpr_udf, inputs)


@udf.jit
def udf_sum(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = lib.fsum(x[i], x[i])
    return 0

@udf.jit
def udf_mult(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = lib2.fmult(x[i], x[i])
    return 0

@udf.jit
def udf_scalar(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1), y: udf.float64):
    for i in range(out.shape[0]):
        out[i] = lib.fsum(x[i], y)
    return 0

@udf.jit
def udf_const_sum(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = 2 * lib.fsum(x[i], x[i])
    return 0

@udf.jit
def udf_const_mult(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = 2 + lib2.fmult(x[i], x[i])
    return 0

@udf.jit
def udf_sin(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = 2.0 + math.sin(x[i]) + lib2.fmult(x[i], x[i])
    return 0

@udf.jit
def udf_sin_cos(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = 2.0 * (math.sin(x[i]) + math.cos(x[i])) + lib2.fmult(x[i], x[i])
    return 0

@udf.jit
def udf_mult_sum(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = 2 + lib2.fmult(x[i], x[i]) * lib.fsum(x[i], x[i])
    return 0

@udf.jit
def udf_mult_sum_mult(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = 2 + lib2.fmult(x[i], x[i]) * lib.fsum(lib2.fmult(x[i], x[i]), x[i])
    return 0

@udf.jit
def udf_mult_sum_mult_scalar(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1), y: udf.float64):
    for i in range(out.shape[0]):
        out[i] = lib2.fmult(x[i], y) * lib.fsum(lib2.fmult(x[i], y), y)
    return 0

@pytest.mark.parametrize(
    "sexpr, func, inputs",
    [
        ("x + x", udf_sum, {"x": ia.arange((10,))}),
        ("x * x", udf_mult, {"x": ia.arange((10,))}),
        ("x + y", udf_scalar, {"x": ia.arange((10,)), "y": 1}),  # scalar as param!
        ("2 * (x + x)", udf_const_sum, {"x": ia.arange((10,))}),
        ("2 + x * x", udf_const_mult, {"x": ia.arange((10,))}),
        ("2 + sin(x) + x * x", udf_sin, {"x": ia.arange((10,))}),
        ("2 * (sin(x) + cos(x)) + x * x", udf_sin_cos, {"x": ia.arange((10,))}),
        ("2 + x * x * (x + x)", udf_mult_sum, {"x": ia.arange((10,))}),
        ("2 + x * x * ((x * x) + x)", udf_mult_sum_mult, {"x": ia.arange((10,))}),
        ("x * y * ((x * y) + y)", udf_mult_sum_mult_scalar, {"x": ia.arange((10,)), "y": 2}),
    ]
)
def test_udf(sexpr, func, inputs):
    expr = ia.expr_from_string(sexpr, inputs)
    out = expr.eval()

    user_inputs = []
    user_params = []
    for value in inputs.values():
        if isinstance(value, ia.IArray):
            user_inputs.append(value)
        else:
            user_params.append(value)

    expr_udf = ia.expr_from_udf(func, user_inputs, user_params)
    out_udf = expr_udf.eval()

    tol = 1e-14
    np.testing.assert_allclose(out.data, out_udf.data, rtol=tol, atol=tol)
