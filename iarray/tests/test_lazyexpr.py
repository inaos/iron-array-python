import pytest
import iarray as ia
import numpy as np


# Different operand fusions inside expressions
@pytest.mark.parametrize(
    "expr, np_expr",
    [
        # Simple expressions
        (
            "(x + y) + z",
            "(x + y) + z",
        ),
        # Transcendental functions
        (
            "x.cos() + y",
            "np.cos(x) + y",
        ),
        (
            "ia.cos(x) + y",
            "np.cos(x) + y"
        ),
        # Reductions
        (
            "x.sum(axis=1) - 1.35",
            "x.sum(axis=1) - 1.35",
        ),
        (
            "ia.sum(x, axis=1) - 1.35",
            "np.sum(x, axis=1) - 1.35",
        ),
        (
            "1.35 + x.max(axis=1)",
            "1.35 + x.max(axis=1)",
        ),
        (
            "x.mean(axis=0) - y.mean(axis=0)",
            "x.mean(axis=0) - y.mean(axis=0)",
        ),
        # Extended slicing (np.where flavored)
        (
            "x[y < 30]",
            "np.where(y < 30, x, np.nan)",
        ),
        (
            "x[y != 30]",
            "np.where(y != 30, x, np.nan)",
        ),
        (
            "x[y >= 30]",
            "np.where(y >= 30, x, np.nan)",
        ),
        (
            "x[y == 30]",
            "np.where(y == 30, x, np.nan)",
        ),
        (
            "x[y == z]",
            "np.where(y == z, x, np.nan)",
        ),
        (
            "x[y == y]",
            "np.where(y == y, x, np.nan)",
        ),
        (
            "x[(y == 3) & (z == 4)]",
            "np.where((y == 3) & (z == 4), x, np.nan)",
        ),
        (
            "x[(y == 3) | (z == 4)]",
            "np.where((y == 3) | (z == 4), x, np.nan)",
        ),
        (
            "x[((y == 3) & (z == 4)) | (x == 0)]",
            "np.where(((y == 3) & (z == 4)) | (x == 0), x, np.nan)",
        ),
        (
            "x[((y == 3) & ~(z == 4))]",
            "np.where(((y == 3) & ~(z == 4)), x, np.nan)",
        ),
        (
            "x[((y == 3) & (z == 4)) | ~(x == 0)]",
            "np.where(((y == 3) & (z == 4)) | ~(x == 0), x, np.nan)",
        ),
        (
            "x[~(z >= 4)]",
            "np.where(~(z >= 4), x, np.nan)",
        ),
        (
            "x[y != 30] * z",
            "np.where(y != 30, x, np.nan) * z",
        ),
        (
            "x[y != 30] * y[z < 30]",
            "np.where(y != 30, x, np.nan) * np.where(z < 30, y, np.nan)",
        ),
        # Reductions with sliced views
        (
            "(x.min(axis=1) - 1.35) *  y[:,1]",
            "(x.min(axis=1) - 1.35) *  y[:,1]",
        ),
        (
            "(x.prod(axis=1) - 1.35) *  ia.cos(y[:,1])",
            "(x.prod(axis=1) - 1.35) *  np.cos(y[:,1])",
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
    ]
)
def test_expr_fusion(expr, np_expr, dtype):
    shape = [100, 100]
    chunks = [40, 50]
    bshape = [10, 10]

    ia.set_config_defaults(chunks=chunks, blocks=bshape, dtype=dtype)

    x = ia.linspace(shape, 0.1, 0.9)
    y = ia.linspace(shape, 0.5, 1)
    z = ia.linspace(shape, 0.1, 0.9)
    t = ia.linspace(shape, 0.5, 1)

    # NumPy computation
    npx = x.data
    npy = y.data
    npz = z.data
    npt = t.data
    npout = eval("%s" % np_expr, {"np": np, "x": npx, "y": npy, "z": npz, "t": npt})

    # High-level ironarray eval
    lazy_expr = eval(expr, {"ia": ia, "x": x, "y": y, "z": z, "t": t})
    iout2 = lazy_expr.eval()
    npout2 = ia.iarray2numpy(iout2)

    tol = 1e-5 if dtype is np.float32 else 1e-14
    np.testing.assert_allclose(npout, npout2, rtol=tol, atol=tol)
