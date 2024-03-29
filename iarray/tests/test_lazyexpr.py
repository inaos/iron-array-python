import pytest
import iarray as ia
from iarray import udf
import numpy as np


@udf.scalar()  # FIXME: be able to not use the empty ()
def clip(a: udf.float32, amin: udf.float32, amax: udf.float32) -> udf.float32:
    if a < amin:
        return amin
    if a > amax:
        return amax
    return a


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
            "ia.cos(x) + y",
            "np.cos(x) + y",
        ),
        ("ia.cos(x) + y", "np.cos(x) + y"),
        # Reductions
        (
            "ia.sum(x, axis=1) - 1.35",
            "np.sum(x, axis=1) - 1.35",
        ),
        (
            "1.35 + ia.max(x, axis=1)",
            "1.35 + x.max(axis=1)",
        ),
        (
            "ia.mean(x, axis=0) - ia.mean(y, axis=0)",
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
            "(ia.min(x, axis=1) - 1.35) *  y[:,1]",
            "(x.min(axis=1) - 1.35) *  y[:,1]",
        ),
        (
            "(ia.prod(x, axis=1) - 1.35) *  ia.cos(y[:,1])",
            "(x.prod(axis=1) - 1.35) *  np.cos(y[:,1])",
        ),
        # Call scalar UDFs
        (
            "ia.ulib.clip(x, 4, 13)",
            "np.clip(x, 4, 13)",
        ),
        (
            "ia.ulib.clip(x, 4, 13) * z - 1.35",
            "np.clip(x, 4, 13) * z - 1.35",
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
    ],
)
def test_expr_fusion(expr, np_expr, dtype):
    shape = [100, 100]
    chunks = [40, 50]
    bshape = [10, 10]

    ia.set_config_defaults(chunks=chunks, blocks=bshape, dtype=dtype)

    x = ia.linspace(0.1, 0.9, int(np.prod(shape)), shape=shape)
    y = ia.linspace(0.5, 1, int(np.prod(shape)), shape=shape)
    z = ia.linspace(0.1, 0.9, int(np.prod(shape)), shape=shape)
    t = ia.linspace(0.5, 1, int(np.prod(shape)), shape=shape)

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
