import pytest
import iarray as ia
import numpy as np


# Rand
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], "double"),
                             ([4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             ([10, 12, 5], None, "double"),
                             ([4, 3, 5, 2], None, "float")
                         ])
def test_rand(shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_rand(ia.dtshape(shape, pshape, dtype))

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.rand(size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)


# Randn
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], "double"),
                             ([4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             ([10, 12, 5], None, "double"),
                             ([4, 3, 5, 2], None, "float")
                         ])
def test_randn(shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_randn(ia.dtshape(shape, pshape, dtype))

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.randn(size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)


# Beta
@pytest.mark.parametrize("alpha, beta, shape, pshape, dtype",
                         [
                             (3, 4, [10, 12, 5], [2, 3, 2], "double"),
                             (0.1, 5, [4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             (3, 0.2, [10, 12, 5], None, "double"),
                             (0.5, 0.05, [4, 3, 5, 2], None, "float")
                         ])
def test_beta(alpha, beta, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_beta(ia.dtshape(shape, pshape, dtype), alpha, beta)

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.beta(alpha, beta, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)


# Lognormal
@pytest.mark.parametrize("mu, sigma, shape, pshape, dtype",
                         [
                             (3, 4, [10, 12, 5], [2, 3, 2], "double"),
                             (0.1, 5, [4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             (3, 0.2, [10, 12, 5], None, "double"),
                             (0.5, 0.05, [4, 3, 5, 2], None, "float")
                         ])
def test_lognormal(mu, sigma, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_lognormal(ia.dtshape(shape, pshape, dtype), mu, sigma)

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.lognormal(mu, sigma, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)


# Exponential
@pytest.mark.parametrize("beta, shape, pshape, dtype",
                         [
                             (3, [10, 12, 5], [2, 3, 2], "double"),
                             (0.1, [4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             (3, [10, 12, 5], None, "double"),
                             (0.5, [4, 3, 5, 2], None, "float")
                         ])
def test_exponential(beta, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_exponential(ia.dtshape(shape, pshape, dtype), beta)

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.exponential(beta, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)


# Uniform
@pytest.mark.parametrize("a_, b_, shape, pshape, dtype",
                         [
                             (3, 5, [10, 12, 5], [2, 3, 2], "double"),
                             (0.1, 0.2, [4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             (-3, -2, [10, 12, 5], None, "double"),
                             (0.5, 1000, [4, 3, 5, 2], None, "float")
                         ])
def test_uniform(a_, b_, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_uniform(ia.dtshape(shape, pshape, dtype), a_, b_)

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.uniform(a_, b_, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)


# Normal
@pytest.mark.parametrize("mu, sigma, shape, pshape, dtype",
                         [
                             (3, 5, [10, 12, 5], [2, 3, 2], "double"),
                             (0.1, 0.2, [4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             (-3, 2, [10, 12, 5], None, "double"),
                             (0.5, 1000, [4, 3, 5, 2], None, "float")
                         ])
def test_normal(mu, sigma, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_normal(ia.dtshape(shape, pshape, dtype), mu, sigma)

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.normal(mu, sigma, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)


# Bernoulli (compare against np.random.binomial)
@pytest.mark.parametrize("p, shape, pshape, dtype",
                         [
                             (0.7, [10, 12, 5], [2, 3, 2], "double"),
                             (0.01, [4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             (0.15, [10, 12, 5], None, "double"),
                             (0.6, [4, 3, 5, 2], None, "float")
                         ])
def test_bernoulli(p, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_bernoulli(ia.dtshape(shape, pshape, dtype), p)

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.binomial(1, p, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)


# Binomial
@pytest.mark.parametrize("n, p, shape, pshape, dtype",
                         [
                             (3, 0.7, [10, 12, 5], [2, 3, 2], "double"),
                             (10, 0.01, [4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             (1000, 0.15, [10, 12, 5], None, "double"),
                             (5, 0.6, [4, 3, 5, 2], None, "float")
                         ])
def test_binomial(n, p, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_binomial(ia.dtshape(shape, pshape, dtype), n, p)

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.binomial(n, p, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)


# Poisson
@pytest.mark.parametrize("lamb, shape, pshape, dtype",
                         [
                             (3, [10, 12, 5], [2, 3, 2], "double"),
                             (0.01, [4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             (0.15, [10, 12, 5], None, "double"),
                             (5, [4, 3, 5, 2], None, "float")
                         ])
def test_poisson(lamb, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.random_poisson(ia.dtshape(shape, pshape, dtype), lamb)

    npdtype = np.float64 if dtype == "double" else np.float32
    b = np.random.poisson(lamb, size).reshape(shape).astype(npdtype)
    c = ia.numpy2iarray(b)

    assert ia.random_kstest(a, c)
