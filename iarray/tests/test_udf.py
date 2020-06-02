import functools
import math

import numpy as np
import pytest

import iarray as ia
from iarray import udf
from iarray.py2llvm import float64, int64


def cmp_udf_np(f, start, stop, shape, pshape, dtype, cparams):
    """Helper function that compares UDF against numpy.

    Constraints:

    - Input is always 1 linspace array, defined by start and stop
    - Function results do not depend on pshape
    """
    x = ia.linspace(ia.dtshape(shape, pshape, dtype), start, stop, **cparams)
    expr = f.create_expr([x], ia.dtshape(shape, pshape, dtype), **cparams)
    out = expr.eval()

    num = functools.reduce(lambda x, y: x * y, shape)
    x_ref = np.linspace(start, stop, num, dtype=dtype).reshape(shape)
    out_ref = np.empty(num, dtype=dtype).reshape(shape)
    f.py_function(out_ref, x_ref)

    ia.cmp_arrays(out, out_ref)


def cmp_udf_np_strict(f, start, stop, shape, pshape, dtype, cparams):
    """Same as cmp_udf_np but the comparison is done strictly. This is to say:
    numpy arrays are evaluated chunk by chunk, this way it works even when the
    function accesses elements other than the current element.

    Contraints:
    - Input is always 1 linspace array, defined by start and stop
    - Only works for 1 dimension arrays
    """
    assert len(pshape) == 1

    x = ia.linspace(ia.dtshape(shape, pshape, dtype), start, stop, **cparams)
    expr = f.create_expr([x], ia.dtshape(shape, pshape, dtype), **cparams)
    out = expr.eval()

    num = functools.reduce(lambda x, y: x * y, shape)
    x_ref = np.linspace(start, stop, num, dtype=dtype).reshape(shape)
    out_ref = np.empty(num, dtype=dtype).reshape(shape)
    indices = range(0, num, pshape[0])
    for out_ref_slice, x_ref_slice in zip(np.array_split(out_ref, indices), np.array_split(x_ref, indices)):
        f.py_function(out_ref_slice, x_ref_slice)

    ia.cmp_arrays(out, out_ref)


@udf.jit
def f_1dim(out: udf.Array(float64, 1), x: udf.Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        if i % 4 == 0:
            out[i] = 0.0
        elif x[i] > 1.0 or x[i] <= 3.0 and i % 2 == 0:
            out[i] = (math.sin(x[i]) + 1.35) * (x[i] + 4.45) * (x[i] + 8.5)
        else:
            out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


@pytest.mark.parametrize('f', [f_1dim])
def test_1dim(f):
    shape = [20 * 1000]
    pshape = [4 * 1000]
    dtype = np.float64
    cparams = dict(clib=ia.LZ4, clevel=5, nthreads=16)
    start, stop = 0, 10

    cmp_udf_np(f, start, stop, shape, pshape, dtype, cparams)


@udf.jit
def f_2dim(out: udf.Array(float64, 2), x: udf.Array(float64, 2)) -> int64:
    n = x.shape[0]
    m = x.shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = (math.sin(x[i, j]) - 1.35) * (x[i, j] - 4.45) * (x[i, j] - 8.5)

    return 0


@pytest.mark.parametrize('f', [f_2dim])
def test_2dim(f):
    shape = [4000, 800]
    pshape = [1000, 200]
    dtype = np.float64
    blocksize = functools.reduce(lambda x, y: x * y, pshape) * dtype(0).itemsize
    cparams = dict(clib=ia.LZ4, clevel=5, blocksize=blocksize)
    start, stop = 0, 10

    cmp_udf_np(f, start, stop, shape, pshape, dtype, cparams)


@udf.jit
def f_avg(out: udf.Array(float64, 1), x: udf.Array(float64, 1)) -> int64:
    n = x.shape[0]
    for i in range(n):
        value = x[i]
        value += x[i - 1] if i > 0 else x[i]
        value += x[i + 1] if i < n - 1 else x[i]
        out[i] = value / 3

    return 0


@pytest.mark.parametrize('f', [f_avg])
def test_avg(f):
    shape = [100]
    pshape = [6]
    dtype = np.float64
    blocksize = functools.reduce(lambda x, y: x * y, pshape) * dtype(0).itemsize
    cparams = dict(clib=ia.LZ4, clevel=5, blocksize=blocksize)
    start, stop = 0, 10

    cmp_udf_np_strict(f, start, stop, shape, pshape, dtype, cparams)


#
# The tests in this section show up bugs, and so they're skipped for now.
#


@udf.jit
def f_error_bug(out: udf.Array(float64, 1), x: udf.Array(float64, 1)) -> int64:
    n = out.shape[1]
    for i in range(n):
        out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0

@udf.jit
def f_error_user(out: udf.Array(float64, 1), x: udf.Array(float64, 1)) -> int64:
    return 1

@pytest.mark.skip('The test pass, but there is a segfault on finalization')
@pytest.mark.parametrize('f', [f_error_bug, f_error_user])
def test_error(f):
    shape = [20 * 1000]
    pshape = [4 * 1000]
    dtype = np.float64
    cparams = dict(clib=ia.LZ4, clevel=5, nthreads=16)
    start, stop = 0, 10

    x = ia.linspace(ia.dtshape(shape, pshape, dtype), start, stop, **cparams)
    expr = f.create_expr([x], ia.dtshape(shape, pshape, dtype), **cparams)
    with pytest.raises(ValueError) as excinfo:
        out = expr.eval()

    assert "Error in evaluating expr: user_defined_function" in str(excinfo.value)
