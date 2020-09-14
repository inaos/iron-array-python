import functools
import math

import numpy as np
import pytest

import iarray as ia
from iarray import udf
from iarray.udf import int32


def cmp_udf_np(f, start, stop, shape, chunkshape, blockshape, dtype, cparams):
    """Helper function that compares UDF against numpy.

    Constraints:

    - Input is always 1 linspace array, defined by start and stop
    - Function results do not depend on chunkshape/blockshape
    """

    storage = ia.StorageProperties("blosc", chunkshape, blockshape)
    x = ia.linspace(ia.dtshape(shape, dtype), start, stop, storage=storage, **cparams)
    expr = f.create_expr([x], ia.dtshape(shape, dtype), storage=storage, **cparams)
    out = expr.eval()

    num = functools.reduce(lambda x, y: x * y, shape)
    x_ref = np.linspace(start, stop, num, dtype=dtype).reshape(shape)
    out_ref = np.empty(num, dtype=dtype).reshape(shape)
    f.py_function(out_ref, x_ref)

    ia.cmp_arrays(out, out_ref)


def cmp_udf_np_strict(f, start, stop, shape, chunkshape, blockshape, dtype, cparams):
    """Same as cmp_udf_np but the comparison is done strictly. This is to say:
    numpy arrays are evaluated chunk by chunk, this way it works even when the
    function accesses elements other than the current element.

    Contraints:
    - Input is always 1 linspace array, defined by start and stop
    - Only works for 1 dimension arrays
    """
    assert len(chunkshape) == 1
    assert len(blockshape) == 1

    storage = ia.StorageProperties("blosc", chunkshape, blockshape)
    x = ia.linspace(ia.dtshape(shape, dtype), start, stop, storage=storage, **cparams)
    expr = f.create_expr([x], ia.dtshape(shape, dtype), storage=storage, **cparams)
    out = expr.eval()

    num = functools.reduce(lambda x, y: x * y, shape)
    x_ref = np.linspace(start, stop, num, dtype=dtype).reshape(shape)
    out_ref = np.empty(num, dtype=dtype).reshape(shape)
    indices = range(0, num, blockshape[0])
    for out_ref_slice, x_ref_slice in zip(np.array_split(out_ref, indices), np.array_split(x_ref, indices)):
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


@pytest.mark.parametrize('f', [f_1dim])
def test_1dim(f):
    shape = [10 * 1000]
    chunkshape = [3 * 1000]
    blockshape = [3 * 100]
    dtype = np.float64
    cparams = dict(clib=ia.LZ4, clevel=5, nthreads=16)
    start, stop = 0, 10

    cmp_udf_np(f, start, stop, shape, chunkshape, blockshape, dtype, cparams)

    # For the test function to return the same output as the Python function
    # the partition size must be multiple of 3. This is just an example of
    # how the result is not always the same as in the Python function.
    blockshape = [4 * 100]
    with pytest.raises(AssertionError):
        cmp_udf_np(f, start, stop, shape, chunkshape, blockshape, dtype, cparams)


@udf.jit
def f_2dim(out: udf.Array(udf.float64, 2), x: udf.Array(udf.float64, 2)):
    n = x.shape[0]
    m = x.shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = (math.sin(x[i, j]) - 1.35) * (x[i, j] - 4.45) * (x[i, j] - 8.5)

    return 0


@pytest.mark.parametrize('f', [f_2dim])
def test_2dim(f):
    shape = [400, 800]
    chunkshape = [60, 200]
    blockshape = [11, 200]
    dtype = np.float64
    cparams = dict(clib=ia.LZ4, clevel=5)
    start, stop = 0, 10

    cmp_udf_np(f, start, stop, shape, chunkshape, blockshape, dtype, cparams)


@udf.jit
def f_while(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = x.shape[0]
    i: int32 = 0
    while i < n:
        out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
        i = i + 1

    return 0


@pytest.mark.parametrize('f', [f_while])
def test_while(f):
    shape = [2000]
    chunkshape = [1000]
    blockshape = [300]
    dtype = np.float64
    cparams = dict(clib=ia.LZ4, clevel=5)
    start, stop = 0, 10

    cmp_udf_np(f, start, stop, shape, chunkshape, blockshape, dtype, cparams)


@udf.jit
def f_avg(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = x.shape[0]
    for i in range(n):
        value = x[i]
        value += x[i - 1] if i > 0 else x[i]
        value += x[i + 1] if i < n - 1 else x[i]
        out[i] = value / 3

    return 0


@pytest.mark.parametrize('f', [f_avg])
def test_avg(f):
    shape = [1000]
    chunkshape = [300]
    blockshape = [100]
    dtype = np.float64
    cparams = dict(clib=ia.LZ4, clevel=5)
    start, stop = 0, 10

    cmp_udf_np_strict(f, start, stop, shape, chunkshape, blockshape, dtype, cparams)


@udf.jit
def f_error_bug(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    n = out.shape[1]
    for i in range(n):
        out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


@udf.jit
def f_error_user(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    return 1


@pytest.mark.parametrize('f', [f_error_bug, f_error_user])
def test_error(f):
    shape = [20 * 1000]
    chunkshape = [4 * 1000]
    blockshape = [1 * 1000]
    dtype = np.float64
    cparams = dict(clib=ia.LZ4, clevel=5, nthreads=1)
    start, stop = 0, 10

    storage = ia.StorageProperties("blosc", chunkshape, blockshape)
    x = ia.linspace(ia.dtshape(shape, dtype), start, stop, storage=storage, **cparams)
    expr = f.create_expr([x], ia.dtshape(shape, dtype), storage=storage, **cparams)

    with pytest.raises(RuntimeError) as excinfo:
        expr.eval()
    assert "Error in evaluating expr: user_defined_function" in str(excinfo.value)


def f_unsupported_function(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    math.sqrt(5.0)
    return 0


def f_bad_argument_count(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1)):
    math.pow(5)
    return 0


def test_function_call_errors():
    with pytest.raises(TypeError):
        udf.jit(f_unsupported_function)

    with pytest.raises(TypeError):
        udf.jit(f_bad_argument_count)
