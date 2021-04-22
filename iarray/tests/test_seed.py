# This scripts generate the random distributions for testing purposes.

# You can copy this tests to iarray by hand:
# $ cp test_*.iarray $IARRAY_DIR/tests/data/

import pytest
import iarray as ia
import numpy as np


# Rand
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    [
        ([20, 20, 20], [10, 12, 5], [2, 3, 2], np.float64, 0),
        ([12, 31, 11, 22], [4, 3, 5, 2], [2, 2, 2, 2], np.float32, 12),
        ([10, 12, 5], None, None, np.float64, 34567865),
        ([4, 3, 5, 2], None, None, np.float32, 24356),
    ],
)
def test_rand(shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.random_sample(shape, store=store, seed=seed, dtype=dtype)
    b = ia.random.random_sample(shape, store=store, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))

    # Check that the default seed is None
    c = ia.random.random_sample(shape, store=store, dtype=dtype)
    assert np.alltrue(ia.iarray2numpy(b) != ia.iarray2numpy(c))


# Randn
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    [
        ([20, 20, 20], [10, 12, 5], [2, 3, 2], np.float64, 23),
        ([10, 10, 8, 10], [4, 3, 5, 2], [2, 2, 2, 2], np.float32, 1),
        ([10, 12, 5], None, None, np.float64, 1234),
        ([4, 3, 5, 2], None, None, np.float32, 21),
    ],
)
def test_randn(shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.standard_normal(shape, store=store, seed=seed, dtype=dtype)
    b = ia.random.standard_normal(shape, store=store, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Beta
@pytest.mark.parametrize(
    "alpha, beta, shape, chunks, blocks, dtype, seed",
    [
        (3, 4, [20, 20, 30], [10, 12, 5], [2, 3, 4], np.float64, 234),
        (0.1, 5, [12, 13, 8, 7], [4, 3, 5, 2], [2, 2, 5, 2], np.float32, 4),
        (3, 0.2, [10, 12, 5], None, None, np.float64, 567),
        (0.5, 0.05, [4, 3, 5, 2], None, None, np.float32, 33),
    ],
)
def test_beta(alpha, beta, shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.beta(shape, alpha, beta, store=store, seed=seed, dtype=dtype)
    b = ia.random.beta(shape, alpha, beta, store=store, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Lognormal
@pytest.mark.parametrize(
    "mu, sigma, shape, chunks, blocks, dtype, seed",
    [
        (3, 4, [20, 20, 20], [10, 12, 5], [2, 3, 2], np.float64, 4321),
        (0.1, 5, [10, 20, 10, 20], [4, 3, 5, 2], [2, 2, 2, 2], np.float32, 12),
        (3, 0.2, [10, 12, 5], None, None, np.float64, 555),
        (0.5, 0.05, [4, 3, 5, 2], None, None, np.float32, 10234),
    ],
)
def test_lognormal(mu, sigma, shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.lognormal(shape, mu, sigma, store=store, seed=seed, dtype=dtype)
    b = ia.random.lognormal(shape, mu, sigma, store=store, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Exponential
@pytest.mark.parametrize(
    "beta, shape, chunks, blocks, dtype, seed",
    [
        (3, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64, 234),
        (0.1, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32, 43),
        (3, [10, 12, 5], None, None, np.float64, 23456),
        (0.5, [4, 3, 5, 2], None, None, np.float32, 9274),
    ],
)
def test_exponential(beta, shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.exponential(shape, beta, store=store, seed=seed, dtype=dtype)
    b = ia.random.exponential(shape, beta, store=store, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Uniform
@pytest.mark.parametrize(
    "a_, b_, shape, chunks, blocks, dtype, seed",
    [
        (3, 5, [20, 20, 20], [10, 12, 10], [2, 3, 2], np.float64, 1),
        (0.1, 0.2, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32, 4),
        (-3, -2, [10, 12, 5], None, None, np.float64, 3),
        (0.5, 1000, [4, 3, 5, 2], None, None, np.float32, 21),
    ],
)
def test_uniform(a_, b_, shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.uniform(shape, a_, b_, store=store, seed=seed, dtype=dtype)
    b = ia.random.uniform(shape, a_, b_, store=store, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Normal
@pytest.mark.parametrize(
    "mu, sigma, shape, chunks, blocks, dtype, seed",
    [
        (3, 5, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64, 31),
        (0.1, 0.2, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32, 5),
        (-3, 2, [10, 12, 5], None, None, np.float64, 22345),
        (0.5, 1000, [4, 3, 5, 2], None, None, np.float32, 674),
    ],
)
def test_normal(mu, sigma, shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.normal(shape, mu, sigma, store=store, seed=seed, dtype=dtype)
    b = ia.random.normal(shape, mu, sigma, store=store, seed=seed, dtype=dtype)

    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Bernoulli (compare against np.random.binomial)
@pytest.mark.parametrize(
    "p, shape, chunks, blocks, dtype, seed",
    [
        (0.7, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64, 589363),
        (0.01, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32, 357),
        (0.15, [10, 12, 5], None, None, np.float64, 3565279),
        (0.6, [4, 3, 5, 2], None, None, np.float32, 5674),
    ],
)
def test_bernoulli(p, shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.bernoulli(shape, p, store=store, seed=seed, dtype=dtype)
    b = ia.random.bernoulli(shape, p, store=store, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Binomial
@pytest.mark.parametrize(
    "n, p, shape, chunks, blocks, dtype, seed",
    [
        (3, 0.7, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64, 31588),
        (10, 0.01, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32, 3),
        (1000, 0.15, [10, 12, 5], None, None, np.float64, 4563933),
        (5, 0.6, [4, 3, 5, 2], None, None, np.float32, 24726),
    ],
)
def test_binomial(n, p, shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.binomial(shape, n, p, store=store, seed=seed, dtype=dtype)
    b = ia.random.binomial(shape, n, p, store=store, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Poisson
@pytest.mark.parametrize(
    "lamb, shape, chunks, blocks, dtype, seed",
    [
        (3, [20, 20, 12], [10, 12, 5], [3, 5, 2], np.float64, 2345333),
        (0.01, [10, 20, 20, 10], [4, 10, 8, 2], [2, 5, 3, 2], np.float32, 44),
        (0.15, [10, 12, 5], None, None, np.float64, 525),
        (5, [4, 3, 5, 2], None, None, np.float32, 3263),
    ],
)
def test_poisson(lamb, shape, chunks, blocks, dtype, seed):
    if chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)

    a = ia.random.poisson(shape, lamb, store=store, seed=seed, dtype=dtype)
    b = ia.random.poisson(shape, lamb, store=store, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))
