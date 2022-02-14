import functools
import math

import numpy as np
import pytest

import iarray as ia
from iarray import udf
from iarray.udf import int32


def cmp_udf_np(f, start_stop, shape, partitions, dtype, cparams, f_np=None, user_params=None):
    """Helper function that compares UDF against numpy.

    Parameters:
        f          : The User-Defined-Function.
        start_stop : Defines the input arrays, may be a tuple or a list of
                     tuples. Each tuple has 2 elements with the start and stop
                     arguments that define a linspace array.
        partitions : A tuple with the chunk and block shapes for iarrays.
        dtype      : Data type.
        cparams    : Configuration parameters for ironArray.
        f_np       : An equivalent function for NumPy (for incompatible UDFs).
        user_params: user params (scalars)

    Function results must not depend on chunks/blocks, otherwise the
    comparison with numpy will fail.
    """

    if type(start_stop) is tuple:
        start_stop = [start_stop]

    chunks, blocks = partitions
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    inputs = [
        ia.linspace(shape, start, stop, cfg=cfg, dtype=dtype, **cparams)
        for start, stop in start_stop
    ]
    expr = ia.expr_from_udf(f, inputs, user_params, shape=shape, cfg=cfg, **cparams)
    out = expr.eval()

    num = functools.reduce(lambda x, y: x * y, shape)
    out_ref = np.empty(num, dtype=dtype).reshape(shape)
    args = [
        np.linspace(start, stop, num, dtype=dtype).reshape(shape) for start, stop in start_stop
    ]
    if user_params is not None:
        args += user_params
    if f_np is None:
        f.py_function(out_ref, *args)
    else:
        f_np(out_ref, *args)

    ia.cmp_arrays(out, out_ref)


def cmp_udf_np_strict(f, start, stop, shape, partitions, dtype, cparams):
    """Same as cmp_udf_np but the comparison is done strictly. This is to say:
    numpy arrays are evaluated chunk by chunk, this way it works even when the
    function accesses elements other than the current element.

    Contraints:
    - Input is always 1 linspace array, defined by start and stop
    - Only works for 1 dimension arrays
    """
    chunks, blocks = partitions
    assert len(chunks) == 1
    assert len(blocks) == 1
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    x = ia.linspace(shape, start, stop, cfg=cfg, dtype=dtype, **cparams)
    assert x.cfg.dtype == dtype
    # Both functions should work, but we are encouraging ia.expr_from_udf()
    # expr = f.create_expr([x], dtshape, cfg=cfg, **cparams)
    expr = ia.expr_from_udf(f, [x], cfg=cfg, **cparams)

    out = expr.eval()

    num = functools.reduce(lambda x, y: x * y, shape)
    x_ref = np.linspace(start, stop, num, dtype=dtype).reshape(shape)
    out_ref = np.empty(num, dtype=dtype).reshape(shape)
    indices = range(0, num, blocks[0])
    for out_ref_slice, x_ref_slice in zip(
        np.array_split(out_ref, indices), np.array_split(x_ref, indices)
    ):
        f.py_function(out_ref_slice, x_ref_slice)

    ia.cmp_arrays(out, out_ref)


@udf.jit
def f_1dim(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = out.shape[0]
    for i in range(n):
        if i % 3 == 0:
            out[i] = 0.0
        elif x[i] > 1.0 or x[i] <= 3.0 and i % 2 == 0:
            out[i] = (math.sin(x[i]) + 1.35) * (x[i] + 4.45) * (x[i] + 8.5)
        else:
            out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


@pytest.mark.parametrize("f", [f_1dim])
def test_1dim(f):
    shape = [10 * 1000]
    chunks = [3 * 1000]
    blocks = [3 * 100]
    dtype = np.float64
    cparams = dict(nthreads=16)
    start, stop = 0, 10

    cmp_udf_np(f, (start, stop), shape, (chunks, blocks), dtype, cparams)

    # For the test function to return the same output as the Python function
    # the partition size must be multiple of 3. This is just an example of
    # how the result is not always the same as in the Python function.
    blocks = [4 * 100]
    with pytest.raises(AssertionError):
        cmp_udf_np(f, (start, stop), shape, (chunks, blocks), dtype, cparams)


@udf.jit
def f_1dim_f32(out: udf.Array(udf.float32, 1), x: udf.Array(udf.float32, 1)):
    n = out.shape[0]
    for i in range(n):
        if i % 3 == 0:
            out[i] = 0.0
        elif x[i] > 1.0 or x[i] <= 3.0 and i % 2 == 0:
            out[i] = (math.sin(x[i]) + 1.35) * (x[i] + 4.45) * (x[i] + 8.5)
        else:
            out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


@udf.jit
def f_math(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = out.shape[0]
    for i in range(n):
        if x[i] > 0.0:
            out[i] = (
                math.log(x[i])
                + math.log10(x[i])
                + math.sqrt(x[i])
                + math.floor(x[i])
                + math.ceil(x[i])
                + math.fabs(x[i])
            )
        else:
            out[i] = x[i]

    return 0


@udf.jit
def f_2dim(out: udf.Array(udf.float64, 2), x: udf.Array(udf.float64, 2)):
    n = x.shape[0]
    m = x.shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = (math.sin(x[i, j]) - 1.35) * (x[i, j] - 4.45) * (x[i, j] - 8.5)

    return 0


@pytest.mark.parametrize("f", [f_2dim])
def test_2dim(f):
    shape = [400, 800]
    chunks = [60, 200]
    blocks = [11, 200]
    dtype = np.float64
    cparams = dict()
    start, stop = 0, 10

    cmp_udf_np(f, (start, stop), shape, (chunks, blocks), dtype, cparams)


@udf.jit
def f_while(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = x.shape[0]
    i: int32 = 0
    while i < n:
        out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
        i = i + 1

    return 0


@pytest.mark.parametrize("f", [f_while])
def test_while(f):
    shape = [2000]
    chunks = [1000]
    blocks = [300]
    dtype = np.float64
    cparams = dict()
    start, stop = 0, 10

    cmp_udf_np(f, (start, stop), shape, (chunks, blocks), dtype, cparams)


@udf.jit
def f_ifexp(out: udf.Array(udf.float64, 2)):
    n = out.shape[0]
    m = out.shape[1]
    start_n = out.window_start[0]
    start_m = out.window_start[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = 1.0 if i + start_n == j + start_m else 0.0

    return 0


# NumPy counterpart of the above
def f_ifexp_np(out):
    n = out.shape[0]
    m = out.shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = 1.0 if i == j else 0.0

    return 0


@pytest.mark.parametrize("f, f_np", [(f_ifexp, f_ifexp_np)])
def test_ifexp(f, f_np):
    shape = [400, 800]
    chunkshape = [60, 200]
    blockshape = [11, 200]
    dtype = np.float64
    cparams = dict()

    cmp_udf_np(f, [], shape, (chunkshape, blockshape), dtype, cparams, f_np=f_np)


@udf.jit
def f_ifexp2(out: udf.Array(udf.float64, 2)):
    n = out.shape[0]
    m = out.shape[1]
    if UDFJIT:
        start_n = out.window_start[0]
        start_m = out.window_start[1]
    for i in range(n):
        for j in range(m):
            if UDFJIT:
                out[i, j] = 1.0 if i + start_n == j + start_m else 0.0
            else:
                out[i, j] = 1.0 if i == j else 0.0

    return 0


@pytest.mark.parametrize("f", [f_ifexp2])
def test_ifexp2(f):
    shape = [400, 800]
    chunkshape = [60, 200]
    blockshape = [11, 200]
    dtype = np.float64
    cparams = dict()

    cmp_udf_np(f, [], shape, (chunkshape, blockshape), dtype, cparams)


@udf.jit
def f_avg(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = x.shape[0]
    for i in range(n):
        value = x[i]
        value += x[i - 1] if i > 0 else x[i]
        value += x[i + 1] if i < n - 1 else x[i]
        out[i] = value / 3

    return 0


@pytest.mark.parametrize("f", [f_avg])
def test_avg(f):
    shape = [1000]
    chunks = [300]
    blocks = [100]
    dtype = np.float64
    cparams = dict()
    start, stop = 0, 10

    cmp_udf_np_strict(f, start, stop, shape, (chunks, blocks), dtype, cparams)


@udf.jit
def f_error_bug(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = out.shape[1]
    for i in range(n):
        out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


@udf.jit
def f_error_user(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    return 1


@pytest.mark.parametrize("f", [f_error_bug, f_error_user])
def test_error(f):
    shape = [20 * 1000]
    chunks = [4 * 1000]
    blocks = [1 * 1000]
    dtype = np.float64
    cparams = dict(nthreads=1)
    start, stop = 0, 10

    cfg = ia.Config(chunks=chunks, blocks=blocks)
    x = ia.linspace(shape, start, stop, cfg=cfg, dtype=dtype, **cparams)
    expr = f.create_expr([x], cfg=cfg, **cparams)

    try:
        expr.eval()
    except ia.ext.IArrayError:
        pass
    else:
        assert False


def f_unsupported_function(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    math.isinf(5.0)
    return 0


def f_bad_argument_count(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    math.pow(5)
    return 0


def test_function_call_errors():
    with pytest.raises(TypeError):
        udf.jit(f_unsupported_function)

    with pytest.raises(TypeError):
        udf.jit(f_bad_argument_count)


@udf.jit
def f_math2(
    out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1), y: udf.Array(udf.float64, 1)
):
    n = out.shape[0]
    for i in range(n):
        out[i] = math.pow(x[i], y[i]) + math.atan2(x[i], y[i])

    return 0


@pytest.mark.parametrize("f", [f_math2])
def test_math2(f):
    shape = [10 * 1000]
    chunks = [3 * 1000]
    blocks = [3 * 100]
    dtype = np.float64
    cparams = dict(nthreads=16)
    start, stop = 0, 10

    cmp_udf_np(f, [(start, stop), (start, stop)], shape, (chunks, blocks), dtype, cparams)


@udf.jit
def f_user_params(
    out: udf.Array(udf.float64, 1),
    x: udf.Array(udf.float64, 1),
    a: udf.float64,
    b: udf.float64,
):
    n = out.shape[0]
    for i in range(n):
        out[i] = x[i] * a + b

    return 0


@pytest.mark.parametrize("f", [f_user_params])
def test_user_params(f):
    shape = [10 * 1000]
    chunks = [3 * 1000]
    blocks = [3 * 100]
    dtype = np.float64
    cparams = dict(nthreads=16)
    start, stop = 0, 10

    cmp_udf_np(f, (start, stop), shape, (chunks, blocks), dtype, cparams, user_params=[2.5, 1.2])
