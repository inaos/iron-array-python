# This scripts generate the random distributions for testing purposes.

# You can copy this tests to iarray by hand:
# $ cp test_*.iarray $IARRAY_DIR/tests/data/

import pytest
import iarray as ia
import numpy as np

rand_data = [
    ([20, 20, 20], [10, 12, 5], [2, 3, 2], np.float64, 0),
    pytest.param([12, 31, 11, 22], [4, 3, 5, 2], [2, 2, 2, 2], np.float32, 12, marks=pytest.mark.heavy),
    ([10, 12, 5], [5, 6, 5], [5, 3, 5], np.float64, 34567865),
    ([4, 3, 5, 2], [4, 3, 5, 2], [2, 3, 5, 1], np.float32, 24356),
]

# Rand
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_rand(shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.random_sample(shape, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.random_sample(shape, cfg=cfg, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))

    # Check that the default seed is None
    c = ia.random.random_sample(shape, cfg=cfg, dtype=dtype)
    assert np.alltrue(ia.iarray2numpy(b) != ia.iarray2numpy(c))


# Randn
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_randn(shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.standard_normal(shape, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.standard_normal(shape, cfg=cfg, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Beta
@pytest.mark.parametrize(
    "alpha, beta",
    [
        (3, 4),
        (0.1, 5),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_beta(alpha, beta, shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.beta(shape, alpha, beta, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.beta(shape, alpha, beta, cfg=cfg, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Lognormal
@pytest.mark.parametrize(
    "mu, sigma",
    [
        (3, 4),
        (3, 0.2),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_lognormal(mu, sigma, shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.lognormal(shape, mu, sigma, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.lognormal(shape, mu, sigma, cfg=cfg, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Exponential
@pytest.mark.parametrize(
    "beta",
    [
        3,
        0.1,
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_exponential(beta, shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.exponential(shape, beta, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.exponential(shape, beta, cfg=cfg, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Uniform
@pytest.mark.parametrize(
    "a_, b_",
    [
        (-3, -2),
        (0.5, 1000),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_uniform(a_, b_, shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.uniform(shape, a_, b_, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.uniform(shape, a_, b_, cfg=cfg, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Normal
@pytest.mark.parametrize(
    "mu, sigma",
    [
        (0.1, 0.2),
        (-10, 0.01),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_normal(mu, sigma, shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.normal(shape, mu, sigma, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.normal(shape, mu, sigma, cfg=cfg, seed=seed, dtype=dtype)

    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Bernoulli (compare against np.random.binomial)
@pytest.mark.parametrize(
    "p",
    [0.01, 0.15, 0.6],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_bernoulli(p, shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.bernoulli(shape, p, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.bernoulli(shape, p, cfg=cfg, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Binomial
@pytest.mark.parametrize(
    "n, p",
    [
        (10, 0.01),
        (5, 0.6),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_binomial(n, p, shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.binomial(shape, n, p, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.binomial(shape, n, p, cfg=cfg, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))


# Poisson
@pytest.mark.parametrize(
    "lamb",
    [5, 0.15],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype, seed",
    rand_data,
)
def test_poisson(lamb, shape, chunks, blocks, dtype, seed):
    cfg = ia.Config(chunks=chunks, blocks=blocks)

    a = ia.random.poisson(shape, lamb, cfg=cfg, seed=seed, dtype=dtype)
    b = ia.random.poisson(shape, lamb, cfg=cfg, seed=seed, dtype=dtype)
    np.testing.assert_array_equal(ia.iarray2numpy(a), ia.iarray2numpy(b))
