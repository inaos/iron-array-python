import functools
import math

import numpy as np
import pytest

import iarray as ia
from iarray import udf


def cmp_udf_np(
    f,
    start_stop,
    shape,
    chunks,
    blocks,
    dtype,
    cparams,
    input_factory=ia.linspace,
    user_params=None,
    f_np=None,
    np_dtype=None,
):
    """Helper function that compares UDF against numpy.

    Parameters:
        f            : The User-Defined-Function.
        start_stop   : A list of tuples defining the input arrays. Each tuple
                       has 2 elements with the start and stop arguments that
                       define each input array.
        shape        : Shape of the iarrays.
        chunks       : Chunks shape for iarrays.
        blocks       : Blocks shape for iarrays.
        dtype        : Data type.
        cparams      : Configuration parameters for ironArray.
        input_factory: function used to generate the input arrays, by default linspace
        user_params  : User parameters (scalars)
        f_np         : An equivalent function for NumPy (for incompatible UDFs).

    Function results must not depend on chunks/blocks, otherwise the
    comparison with numpy will fail.
    """

    assert type(start_stop) is list

    if f_np is None:
        f_np = f.py_function

    cfg = ia.Config(chunks=chunks, blocks=blocks)

    inputs = [
        input_factory(shape, start, stop, cfg=cfg, dtype=dtype, np_dtype=None, **cparams)
        for start, stop in start_stop
    ]
    expr = ia.expr_from_udf(
        f, inputs, user_params, shape=shape, dtype=dtype, np_dtype=np_dtype, cfg=cfg, **cparams
    )
    out = expr.eval()

    out_dtype = dtype if np_dtype is None else np.dtype(np_dtype)
    num = functools.reduce(lambda x, y: x * y, shape)
    out_ref = np.zeros(num, dtype=out_dtype).reshape(shape)
    args = [x.data for x in inputs]
    if user_params is not None:
        args += user_params

    f_np(out_ref, *args)
    if out_dtype in [np.float64, np.float32]:
        ia.cmp_arrays(out, out_ref)
    else:
        if type(out) is ia.IArray:
            out = ia.iarray2numpy(out)
        if type(out_ref) is ia.IArray:
            out_ref = ia.iarray2numpy(out_ref)
        np.testing.assert_array_equal(out, out_ref)


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
    expr = ia.expr_from_udf(f, [x], cfg=cfg, **cparams)

    out = expr.eval()

    num = functools.reduce(lambda x, y: x * y, shape)
    x_ref = np.linspace(start, stop, num, dtype=dtype).reshape(shape)
    out_ref = np.zeros(num, dtype=dtype).reshape(shape)
    indices = range(0, num, blocks[0])
    for out_ref_slice, x_ref_slice in zip(
        np.array_split(out_ref, indices), np.array_split(x_ref, indices)
    ):
        f.py_function(out_ref_slice, x_ref_slice)

    ia.cmp_arrays(out, out_ref)


@udf.jit()
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


@udf.jit
def f_fabs_copysign(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = out.shape[0]
    for i in range(n):
        out[i] = math.fabs(x[i]) + math.copysign(x[i], -1.0)

    return 0


@udf.jit
def f_while(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = x.shape[0]
    i = 0
    while i < n:
        out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
        i = i + 1

    return 0


@udf.jit
def f_1dim_int(out: udf.Array(udf.int64, 1), x: udf.Array(udf.int64, 1)):
    n = out.shape[0]
    for i in range(n):
        if i % 3 == 0:
            out[i] = 0
        elif x[i] > 1 or x[i] <= 3 and i % 2 == 0:
            out[i] = (x[i] + 4) * (x[i] + 8)
        else:
            out[i] = (x[i] - 4) * (x[i] - 8)

    return 0


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
def f_math_int(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = out.shape[0]
    for i in range(n):
        out[i] = x[i] * math.cos(1)
    return 0


@udf.jit
def f_avg(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = x.shape[0]
    for i in range(n):
        value = x[i]
        value += x[i - 1] if i > 0 else x[i]
        value += x[i + 1] if i < n - 1 else x[i]
        out[i] = value / 3

    return 0


@udf.jit
def f_power(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = 2.71828 ** x[i]
    return 0


@udf.jit
def f_unary_float(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    for i in range(out.shape[0]):
        out[i] = -x[i]
    return 0


@udf.jit
def f_unary_int(out: udf.Array(udf.int64, 1), x: udf.Array(udf.int64, 1)):
    n = out.shape[0]
    for i in range(out.shape[0]):
        out[i] = -x[i]
    return 0


@udf.jit
def f_idx_const(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = out.shape[0]
    for i in range(n):
        out[0] = x[i]

    return 0


@udf.jit
def f_idx_var(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    var = 0
    n = out.shape[0]
    for i in range(n):
        out[var] = x[i]

    return 0


@pytest.mark.parametrize(
    "f, dtype",
    [
        (f_1dim, np.float64),
        (f_fabs_copysign, np.float64),
        (f_while, np.float64),
        (f_1dim_int, np.int64),
        (f_1dim_f32, np.float32),
        (f_math, np.float64),
        (f_math_int, np.float64),
        (f_avg, np.float64),
        # Power
        (f_power, np.float64),
        # Unary operator
        (f_unary_float, np.float64),
        (f_unary_int, np.int64),
        # https://github.com/inaos/iron-array/issues/502
        (f_idx_const, np.float64),
        (f_idx_var, np.float64),
    ],
)
def test_1dim(f, dtype):
    shape = [10 * 1000]
    chunks = [3 * 1000]
    blocks = [3 * 100]
    cparams = dict(nthreads=16)
    start, stop = 0, 100  # to avoid overflows, don't use a too large stop here

    cmp_udf_np_strict(f, start, stop, shape, (chunks, blocks), dtype, cparams)


@pytest.mark.parametrize("f", [f_1dim])
def test_partition_mismatch(f):
    shape = [10 * 1000]
    chunks = [3 * 1000]
    blocks = [3 * 100]
    dtype = np.float64
    cparams = dict(nthreads=16)
    start, stop = 0, 10

    # For the test function to return the same output as the Python function
    # the partition size must be multiple of 3. This is just an example of
    # how the result is not always the same as in the Python function.
    blocks = [4 * 100]
    with pytest.raises(AssertionError):
        cmp_udf_np(f, [(start, stop)], shape, chunks, blocks, dtype, cparams)


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
    shape = [40, 80]  # [400, 800]
    chunks = [6, 20]  # [60, 200]
    blocks = [4, 2]  # [11, 200]
    dtype = np.float64
    cparams = dict()
    start, stop = 0, 10

    cmp_udf_np(f, [(start, stop)], shape, chunks, blocks, dtype, cparams)


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

    cmp_udf_np(f, [], shape, chunkshape, blockshape, dtype, cparams, f_np=f_np)


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

    cmp_udf_np(f, [], shape, chunkshape, blockshape, dtype, cparams)


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

    with pytest.raises(ia.IArrayError):
        expr.eval()


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

    cmp_udf_np(f, [(start, stop), (start, stop)], shape, chunks, blocks, dtype, cparams)


@udf.jit
def f_user_params(
    out: udf.Array(udf.float64, 1),
    x: udf.Array(udf.float64, 1),
    a: udf.float64,
    b: udf.float64,
    divide: udf.bool,
):
    n = out.shape[0]
    for i in range(n):
        if divide:
            out[i] = x[i] / a + b
        else:
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

    user_params = [2.5, 1, True]
    cmp_udf_np(f, [(start, stop)], shape, chunks, blocks, dtype, cparams, user_params=user_params)


@udf.jit
def f_idx_int(out: udf.Array(udf.int64, 1), x: udf.Array(udf.int64, 1)):
    n = out.shape[0]
    for i in range(n):
        out[i] = x[i]

    return 0


@pytest.mark.parametrize("f", [f_idx_int])
def test_idx_var_datetime(f):
    shape = [10 * 1000]
    chunks = [3 * 1000]
    blocks = [3 * 100]
    dtype = np.int64
    cparams = dict(nthreads=16)
    start, stop = 0, 10 * 1000

    cmp_udf_np(
        f,
        [(start, stop)],
        shape,
        chunks,
        blocks,
        dtype,
        cparams,
        input_factory=ia.arange,
        np_dtype="m8[Y]",
    )
