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
        ("x + x", "lib.fsum(x, x)", {"x": ia.arange(5, step=0.5, shape=(10,))}),
        ("x * x", "lib2.fmult(x, x)", {"x": ia.arange(10, shape=(10,))}),
        ("x + y", "lib.fsum(x, y)", {"x": ia.arange(5, step=0.5, shape=(10,)), "y": 1}),  # scalar as param!
        ("2 * (x + x)", "2 * lib.fsum(x, x)", {"x": ia.arange(10, shape=(10,))}),
        ("2 + x * x", "2 + lib2.fmult(x, x)", {"x": ia.arange(5, step=0.5, shape=(10,))}),
        ("2 + sin(x) + x * x", "2 + sin(x) + lib2.fmult(x, x)", {"x": ia.arange(10, shape=(10,))}),
        (
            "2 * (sin(x) + cos(x)) + x * x",
            "2 * (sin(x) + cos(x)) + lib2.fmult(x, x)",
            {"x": ia.arange(5, step=0.5, shape=(10,))},
        ),
        (
            "2 + x * x * (x + x)",
            "2 + lib2.fmult(x, x) * lib.fsum(x, x)",
            {"x": ia.arange(10, shape=(10,))},
        ),
        (
            "2 + x * x * ((x * x) + x)",
            "2 + lib2.fmult(x, x) * lib.fsum(lib2.fmult(x, x), x)",
            {"x": ia.arange(5, step=0.5, shape=[10])},
        ),
        (
            "x * y * ((x * y) + y)",
            "lib2.fmult(x, y) * lib.fsum(lib2.fmult(x, y), y)",
            {"x": ia.arange(10, shape=[10]), "y": 2},
        ),
    ],
)
def test_simple(sexpr, sexpr_udf, inputs):
    assert "lib" in ia.udf_registry
    assert "fsum" in set(fname for fname in ia.udf_registry["lib"])
    assert "fsum" in (f.name for f in ia.udf_registry.iter_funcs("lib"))
    assert "lib.fsum" in (f for f in ia.udf_registry.iter_all_func_names())

    assert "lib2" in ia.udf_registry
    assert {"fmult"} == set(fname for fname in ia.udf_registry["lib2"])
    assert "fmult" in (f.name for f in ia.udf_registry.iter_funcs("lib2"))
    assert "lib2.fmult" in (f for f in ia.udf_registry.iter_all_func_names())
    assert {"lib.fsum", "lib2.fmult"}.issubset(set(ia.udf_registry.iter_all_func_names()))

    expr = ia.expr_from_string(sexpr, inputs)
    expr_udf = ia.expr_from_string(sexpr_udf, inputs)
    out = expr.eval()
    out_udf = expr_udf.eval()
    tol = 1e-14
    np.testing.assert_allclose(out.data, out_udf.data, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "sexpr_udf, inputs",
    [
        ("lib.fsum(x1, x)", {"x": ia.arange(10, shape=(10,))}),
        ("lib.fsum(x1, x1)", {"x": ia.arange(5, step=0.5, shape=(10,))}),
        ("2 + lib.fmult2(x, x)", {"x": ia.arange(10, shape=(10,))}),
        ("2 + lib.fmult(x, x)", {"x": ia.arange(5, step=0.5, shape=(10,))}),
        ("2 + sin(x) + lib.fmult(x, x)", {"x": ia.arange(10, shape=(10,))}),
        ("2 + lib.fmult(x, x) * lib2.fsum(x, x)", {"x": ia.arange(5, step=0.5, shape=(10,))}),
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
        out[i] = 2 + math.sin(x[i]) + lib2.fmult(x[i], x[i])
    return 0


@udf.jit
def udf_sin_cos(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = 2 * (math.sin(x[i]) + math.cos(x[i])) + lib2.fmult(x[i], x[i])
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
def udf_mult_sum_mult_scalar(
    out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1), y: udf.float64
):
    for i in range(out.shape[0]):
        out[i] = lib2.fmult(x[i], y) * lib.fsum(lib2.fmult(x[i], y), y)
    return 0


@pytest.mark.parametrize(
    "sexpr, func, inputs",
    [
        ("x + x", udf_sum, {"x": ia.arange(10, shape=(10,))}),
        ("x * x", udf_mult, {"x": ia.arange(5, step=0.5, shape=(10,))}),
        ("x + y", udf_scalar, {"x": ia.arange(10, shape=(10,)), "y": 1}),  # scalar as param!
        ("2 * (x + x)", udf_const_sum, {"x": ia.arange(5, step=0.5, shape=(10,))}),
        ("2 + x * x", udf_const_mult, {"x": ia.arange(10, shape=(10,))}),
        ("2 + sin(x) + x * x", udf_sin, {"x": ia.arange(5, step=0.5, shape=(10,))}),
        ("2 * (sin(x) + cos(x)) + x * x", udf_sin_cos, {"x": ia.arange(10, shape=(10,))}),
        ("2 + x * x * (x + x)", udf_mult_sum, {"x": ia.arange(5, step=0.5, shape=(10,))}),
        ("2 + x * x * ((x * x) + x)", udf_mult_sum_mult, {"x": ia.arange(10, shape=(10,))}),
        (
            "x * y * ((x * y) + y)",
            udf_mult_sum_mult_scalar,
            {"x": ia.arange(5, step=0.5, shape=(10,)), "y": 2},
        ),
    ],
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


@udf.scalar(lib="lib")
def fcond_none(a: udf.float64, b: udf.float64) -> float:
    if (a + b) > 3:
        x = 1
    else:
        x = 0

    return x


@udf.scalar(lib="lib")
def fcond_both(a: udf.float64, b: udf.float64) -> float:
    if (a + b) > 3:
        return 1
    else:
        return 0


@udf.scalar(lib="lib")
def fcond_if(a: udf.float64, b: udf.float64) -> float:
    if (a + b) > 3:
        return 1
    else:
        x = 0

    return x


@udf.scalar(lib="lib")
def fcond_else(a: udf.float64, b: udf.float64) -> float:
    if (a + b) > 3:
        x = 1
    else:
        return 0

    return x


@pytest.mark.parametrize("name", ["fcond_none", "fcond_both", "fcond_if", "fcond_else"])
def test_conditional(name):
    a = ia.arange(5, step=0.5, shape=[10])
    b = ia.ones([10])
    expr = ia.expr_from_string(f"lib.{name}(a, b)", {"a": a, "b": b})
    out = expr.eval()

    out_ref = (a.data + b.data) > 3
    np.testing.assert_array_equal(out.data, out_ref)


@udf.scalar(lib="lib")
def fcond_f64(a: udf.float64, b: udf.float64) -> udf.float64:
    if (a + b) > 3:
        return 1
    else:
        return 0


@udf.scalar(lib="lib")
def fcond_f32(a: udf.float32, b: udf.float32) -> udf.float32:
    if (a + b) > 3:
        return 1
    else:
        return 0


@udf.scalar(lib="lib")
def fcond_i64(a: udf.int64, b: udf.int64) -> udf.int64:
    if (a + b) > 3:
        return 1
    else:
        return 0


@udf.scalar(lib="lib")
def fcond_i32(a: udf.int32, b: udf.int32) -> udf.int32:
    if (a + b) > 3:
        return 1
    else:
        return 0


@udf.scalar(lib="lib")
def fcond_i16(a: udf.int16, b: udf.int16) -> udf.int16:
    if (a + b) > 3:
        return 1
    else:
        return 0


@udf.scalar(lib="lib")
def fcond_i8(a: udf.int8, b: udf.int8) -> udf.int8:
    if (a + b) > 3:
        return 1
    else:
        return 0


@pytest.mark.parametrize(
    "name, dtype",
    [
        ("fcond_f64", np.float64),
        ("fcond_f32", np.float32),
        ("fcond_i64", np.int64),
        ("fcond_i32", np.int32),
        ("fcond_i16", np.int16),
        ("fcond_i8", np.int8),
    ],
)
def test_types(name, dtype):
    a = ia.arange(10, shape=[10], dtype=dtype)
    b = ia.ones([10], dtype=dtype)
    expr = ia.expr_from_string(f"lib.{name}(a, b)", {"a": a, "b": b})
    out = expr.eval()

    out_ref = (a.data + b.data) > 3
    np.testing.assert_array_equal(out.data, out_ref)
