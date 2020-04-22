# This scripts generate the random distributions for testing purposes.

# You can copy this tests to iarray by hand:
# $ cp test_*.iarray $IARRAY_DIR/tests/data/

import pytest
import iarray as ia
import numpy as np
import inspect

# Rand
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], np.float64),
                             ([4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             ([10, 12, 5], None, np.float64),
                             ([4, 3, 5, 2], None, np.float32)
                         ])
def test_rand(shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_rand(ia.dtshape(shape, pshape, dtype), storage=storage)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.rand(size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f""
    dtype_symbol = "f" if dtype == np.float32 else "d"

    filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"

    if pshape is not None:
        storage.enforce_frame = True
        storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)


# Randn
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], np.float64),
                             ([4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             ([10, 12, 5], None, np.float64),
                             ([4, 3, 5, 2], None, np.float32)
                         ])
def test_randn(shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_randn(ia.dtshape(shape, pshape, dtype),storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.randn(size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f""
    dtype_symbol = "f" if dtype == np.float32 else "d"

    filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"

    if pshape is not None:
        storage.enforce_frame = True
        storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)


# Beta
@pytest.mark.parametrize("alpha, beta, shape, pshape, dtype",
                         [
                             (3, 4, [10, 12, 5], [2, 3, 2], np.float64),
                             (0.1, 5, [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (3, 0.2, [10, 12, 5], None, np.float64),
                             (0.5, 0.05, [4, 3, 5, 2], None, np.float32)
                         ])
def test_beta(alpha, beta, shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_beta(ia.dtshape(shape, pshape, dtype), alpha, beta, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.beta(alpha, beta, size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f"_{str(alpha).replace('.', '')}_{str(beta).replace('.', '')}"
    dtype_symbol = "f" if dtype == np.float32 else "d"

    filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"

    if pshape is not None:
        storage.enforce_frame = True
        storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)


# Lognormal
@pytest.mark.parametrize("mu, sigma, shape, pshape, dtype",
                         [
                             (3, 4, [10, 12, 5], [2, 3, 2], np.float64),
                             (0.1, 5, [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (3, 0.2, [10, 12, 5], None, np.float64),
                             (0.5, 0.05, [4, 3, 5, 2], None, np.float32)
                         ])
def test_lognormal(mu, sigma, shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_lognormal(ia.dtshape(shape, pshape, dtype), mu, sigma, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.lognormal(mu, sigma, size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f"_{str(mu).replace('.', '')}_{str(sigma).replace('.', '')}"
    dtype_symbol = "f" if dtype == np.float32 else "d"

    filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"

    if pshape is not None:
        storage.enforce_frame = True
        storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)


# Exponential
@pytest.mark.parametrize("beta, shape, pshape, dtype",
                         [
                             (3, [10, 12, 5], [2, 3, 2], np.float64),
                             (0.1, [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (3, [10, 12, 5], None, np.float64),
                             (0.5, [4, 3, 5, 2], None, np.float32)
                         ])
def test_exponential(beta, shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_exponential(ia.dtshape(shape, pshape, dtype), beta, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.exponential(beta, size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f"_{str(beta).replace('.', '')}"
    dtype_symbol = "f" if dtype == np.float32 else "d"

    filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"

    if pshape is not None:
        storage.enforce_frame = True
        storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)


# Uniform
@pytest.mark.parametrize("a_, b_, shape, pshape, dtype",
                         [
                             (3, 5, [10, 12, 10], [2, 3, 2], np.float64),
                             (0.1, 0.2, [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (-3, -2, [10, 12, 5], None, np.float64),
                             (0.5, 1000, [4, 3, 5, 2], None, np.float32)
                         ])
def test_uniform(a_, b_, shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_uniform(ia.dtshape(shape, pshape, dtype), a_, b_, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.uniform(a_, b_, size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f"_{str(a_).replace('.', '')}_{str(b_).replace('.', '')}"
    dtype_symbol = "f" if dtype == np.float32 else "d"

    filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"

    if pshape is not None:
        storage.enforce_frame = True
        storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)


# Normal
@pytest.mark.parametrize("mu, sigma, shape, pshape, dtype",
                         [
                             (3, 5, [10, 12, 5], [2, 3, 2], np.float64),
                             (0.1, 0.2, [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (-3, 2, [10, 12, 5], None, np.float64),
                             (0.5, 1000, [4, 3, 5, 2], None, np.float32)
                         ])
def test_normal(mu, sigma, shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_normal(ia.dtshape(shape, pshape, dtype), mu, sigma, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.normal(mu, sigma, size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f"_{str(mu).replace('.', '')}_{str(sigma).replace('.', '')}"
    dtype_symbol = "f" if dtype == np.float32 else "d"

    filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"

    if pshape is not None:
        storage.enforce_frame = True
        storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)


# Bernoulli (compare against np.random.binomial)
@pytest.mark.parametrize("p, shape, pshape, dtype",
                         [
                             (0.7, [10, 12, 5], [2, 3, 2], np.float64),
                             (0.01, [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (0.15, [10, 12, 5], None, np.float64),
                             (0.6, [4, 3, 5, 2], None, np.float32)
                         ])
def test_bernoulli(p, shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_bernoulli(ia.dtshape(shape, pshape, dtype), p, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.binomial(1, p, size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f"_{str(p).replace('.', '')}"
    dtype_symbol = "f" if dtype == np.float32 else "d"

    filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"

    if pshape is not None:
        storage.enforce_frame = True
        storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)


# Binomial
@pytest.mark.parametrize("n, p, shape, pshape, dtype",
                         [
                             (3, 0.7, [10, 12, 5], [2, 3, 2], np.float64),
                             (10, 0.01, [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (1000, 0.15, [10, 12, 5], None, np.float64),
                             (5, 0.6, [4, 3, 5, 2], None, np.float32)
                         ])
def test_binomial(n, p, shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_binomial(ia.dtshape(shape, pshape, dtype), n, p, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.binomial(n, p, size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f"_{str(n).replace('.', '')}_{str(p).replace('.', '')}"
    dtype_symbol = "f" if dtype == np.float32 else "d"

    filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"

    if pshape is not None:
        storage.enforce_frame = True
        storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)


# Poisson
@pytest.mark.parametrize("lamb, shape, pshape, dtype",
                         [
                             (3, [10, 12, 5], [2, 3, 2], np.float64),
                             (0.01, [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (0.15, [10, 12, 5], None, np.float64),
                             (5, [4, 3, 5, 2], None, np.float32)
                         ])
def test_poisson(lamb, shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    size = int(np.prod(shape))
    a = ia.random_poisson(ia.dtshape(shape, pshape, dtype), lamb, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.poisson(lamb, size).reshape(shape).astype(npdtype)

    function_name = inspect.currentframe().f_code.co_name
    extra_args = f"_{str(lamb).replace('.', '')}"
    dtype_symbol = "f" if dtype == np.float32 else "d"

    # filename = f"{function_name}_{dtype_symbol}{extra_args}.iarray"
    # if pshape is not None:
    #     storage.enforce_frame = False
    #     storage.filename = filename

    c = ia.numpy2iarray(b, pshape, storage=storage)

    assert ia.random_kstest(a, c)
