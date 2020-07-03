# This scripts generate the random distributions for testing purposes.

# You can copy this tests to iarray by hand:
# $ cp test_*.iarray $IARRAY_DIR/tests/data/

import pytest
import iarray as ia
import numpy as np


# Rand
@pytest.mark.parametrize("shape, chunkshape, blockshape, dtype",
                         [
                             ([20, 20, 20], [10, 12, 5], [2, 3, 2], np.float64),
                             ([12, 31, 11, 22], [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             ([10, 12, 5], None, None, np.float64),
                             ([4, 3, 5, 2], None, None, np.float32)
                         ])
def test_rand(shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_rand(ia.dtshape(shape, dtype), storage=storage)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.rand(size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)


# Randn
@pytest.mark.parametrize("shape, chunkshape, blockshape, dtype",
                         [
                             ([20, 20, 20], [10, 12, 5], [2, 3, 2], np.float64),
                             ([10, 10, 8, 10], [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             ([10, 12, 5], None, None, np.float64),
                             ([4, 3, 5, 2], None, None, np.float32)
                         ])
def test_randn(shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_randn(ia.dtshape(shape, dtype), storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.randn(size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)


# Beta
@pytest.mark.parametrize("alpha, beta, shape, chunkshape, blockshape, dtype",
                         [
                             (3, 4,  [20, 20, 30], [10, 12, 5], [2, 3, 4], np.float64),
                             (0.1, 5, [12, 13, 8, 7], [4, 3, 5, 2], [2, 2, 5, 2], np.float32),
                             (3, 0.2, [10, 12, 5], None, None, np.float64),
                             (0.5, 0.05, [4, 3, 5, 2], None, None, np.float32)
                         ])
def test_beta(alpha, beta, shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_beta(ia.dtshape(shape, dtype), alpha, beta, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.beta(alpha, beta, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)


# Lognormal
@pytest.mark.parametrize("mu, sigma, shape, chunkshape, blockshape, dtype",
                         [
                             (3, 4, [20, 20, 20], [10, 12, 5], [2, 3, 2], np.float64),
                             (0.1, 5, [10, 20, 10, 20], [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (3, 0.2, [10, 12, 5], None, None, np.float64),
                             (0.5, 0.05, [4, 3, 5, 2], None, None, np.float32)
                         ])
def test_lognormal(mu, sigma, shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_lognormal(ia.dtshape(shape, dtype), mu, sigma, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.lognormal(mu, sigma, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)


# Exponential
@pytest.mark.parametrize("beta, shape, chunkshape, blockshape, dtype",
                         [
                             (3, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64),
                             (0.1, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32),
                             (3, [10, 12, 5], None, None, np.float64),
                             (0.5, [4, 3, 5, 2], None, None, np.float32)
                         ])
def test_exponential(beta, shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_exponential(ia.dtshape(shape, dtype), beta, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.exponential(beta, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)


# Uniform
@pytest.mark.parametrize("a_, b_, shape, chunkshape, blockshape, dtype",
                         [
                             (3, 5, [20, 20, 20], [10, 12, 10], [2, 3, 2], np.float64),
                             (0.1, 0.2, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32),
                             (-3, -2, [10, 12, 5], None, None, np.float64),
                             (0.5, 1000, [4, 3, 5, 2], None, None, np.float32)
                         ])
def test_uniform(a_, b_, shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_uniform(ia.dtshape(shape, dtype), a_, b_, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.uniform(a_, b_, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)


# Normal
@pytest.mark.parametrize("mu, sigma, shape, chunkshape, blockshape, dtype",
                         [
                             (3, 5, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64),
                             (0.1, 0.2, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32),
                             (-3, 2, [10, 12, 5], None, None, np.float64),
                             (0.5, 1000, [4, 3, 5, 2], None, None, np.float32)
                         ])
def test_normal(mu, sigma, shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_normal(ia.dtshape(shape, dtype), mu, sigma, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.normal(mu, sigma, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)


# Bernoulli (compare against np.random.binomial)
@pytest.mark.parametrize("p, shape, chunkshape, blockshape, dtype",
                         [
                             (0.7, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64),
                             (0.01, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32),
                             (0.15, [10, 12, 5], None, None, np.float64),
                             (0.6, [4, 3, 5, 2], None, None, np.float32)
                         ])
def test_bernoulli(p, shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_bernoulli(ia.dtshape(shape, dtype), p, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.binomial(1, p, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)


# Binomial
@pytest.mark.parametrize("n, p, shape, chunkshape, blockshape, dtype",
                         [
                             (3, 0.7, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64),
                             (10, 0.01, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32),
                             (1000, 0.15, [10, 12, 5], None, None, np.float64),
                             (5, 0.6, [4, 3, 5, 2], None, None, np.float32)
                         ])
def test_binomial(n, p, shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_binomial(ia.dtshape(shape, dtype), n, p, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.binomial(n, p, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)


# Poisson
@pytest.mark.parametrize("lamb, shape, chunkshape, blockshape, dtype",
                         [
                             (3, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64),
                             (0.01, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32),
                             (0.15, [10, 12, 5], None, None, np.float64),
                             (5, [4, 3, 5, 2], None, None, np.float32)
                         ])
def test_poisson(lamb, shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)

    size = int(np.prod(shape))
    a = ia.random_poisson(ia.dtshape(shape, dtype), lamb, storage=storage)

    npdtype = np.float64 if dtype == np.float64 else np.float32
    b = np.random.poisson(lamb, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b, storage=storage)

    assert ia.random_kstest(a, c)
