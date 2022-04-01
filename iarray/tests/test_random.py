# This scripts generate the random distributions for testing purposes.

# You can copy this tests to iarray by hand:
# $ cp test_*.iarray $IARRAY_DIR/tests/data/

import pytest
import iarray as ia
import numpy as np

dtype_data1 = [np.float64, np.float32]
dtype_data2 = [np.int32]

rand_data = [
    ([20, 20, 20], [10, 12, 5], [2, 3, 2]),
    ([10, 12, 5], [10, 12, 4], [10, 6, 4]),
    pytest.param([12, 31, 11, 22], [4, 3, 5, 2], [2, 2, 2, 2], marks=pytest.mark.heavy),
    pytest.param([4, 3, 5, 2], [2, 2, 2, 2], [2, 2, 1, 2], marks=pytest.mark.heavy),
]

# Rand
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data1,
)
def test_rand(shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))
    i = 0
    while i < 5:
        a = ia.random.random_sample(shape, cfg=cfg, dtype=dtype)
        b = np.random.random_sample(size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False


# Randn
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data1,
)
def test_randn(shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))

    modes = ["r", "r+", "a", "w", "w-"]
    i = 0
    while i < 5:
        mode = modes[i]
        if mode in ["r", "r+"]:
            with pytest.raises(IOError):
                a = ia.random.standard_normal(shape, cfg=cfg, dtype=dtype, mode=mode)
            mode = "w-"
        a = ia.random.standard_normal(shape, cfg=cfg, dtype=dtype, mode=mode)
        b = np.random.standard_normal(size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False


# Beta
@pytest.mark.parametrize(
    "alpha, beta",
    [
        (3, 4),
        (0.1, 5),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data1,
)
def test_beta(alpha, beta, shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))

    modes = ["r", "r+", "a", "w", "w-"]
    i = 0
    while i < 5:
        mode = modes[i]
        if mode in ["r", "r+"]:
            with pytest.raises(IOError):
                a = ia.random.beta(shape, alpha, beta, cfg=cfg, dtype=dtype, mode=mode)
            mode = "w-"
        a = ia.random.beta(shape, alpha, beta, cfg=cfg, dtype=dtype, mode=mode)
        b = np.random.beta(alpha, beta, size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False


# Lognormal
@pytest.mark.parametrize(
    "mu, sigma",
    [
        (3, 4),
        (3, 0.2),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data1,
)
def test_lognormal(mu, sigma, shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))

    modes = ["r", "r+", "a", "w", "w-"]
    i = 0
    while i < 5:
        mode = modes[i]
        if mode in ["r", "r+"]:
            with pytest.raises(IOError):
                a = ia.random.lognormal(shape, mu, sigma, cfg=cfg, dtype=dtype, mode=mode)
            mode = "a"
        a = ia.random.lognormal(shape, mu, sigma, cfg=cfg, dtype=dtype, mode=mode)
        b = np.random.lognormal(mu, sigma, size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False


# Exponential
@pytest.mark.parametrize(
    "beta",
    [
        3,
        0.1,
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data1,
)
def test_exponential(beta, shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))

    modes = ["r", "r+", "a", "w", "w-"]
    i = 0
    while i < 5:
        mode = modes[i]
        if mode in ["r", "r+"]:
            with pytest.raises(IOError):
                a = ia.random.exponential(shape, beta, cfg=cfg, dtype=dtype, mode=mode)
            mode = "w"
        a = ia.random.exponential(shape, beta, cfg=cfg, dtype=dtype, mode=mode)
        b = np.random.exponential(beta, size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False


# Uniform
@pytest.mark.parametrize(
    "a_, b_",
    [
        (-3, -2),
        (0.5, 1000),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data1,
)
def test_uniform(a_, b_, shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))

    modes = ["r", "r+", "a", "w", "w-"]
    i = 0
    while i < 5:
        mode = modes[i]
        if mode in ["r", "r+"]:
            with pytest.raises(IOError):
                a = ia.random.uniform(shape, a_, b_, cfg=cfg, dtype=dtype, mode=mode)
            mode = "w-"
        a = ia.random.uniform(shape, a_, b_, cfg=cfg, dtype=dtype, mode=mode)
        b = np.random.uniform(a_, b_, size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False


# Normal
@pytest.mark.parametrize(
    "mu, sigma",
    [
        (0.1, 0.2),
        (-10, 0.01),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data1,
)
def test_normal(mu, sigma, shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))

    modes = ["r", "r+", "a", "w", "w-"]
    i = 0
    while i < 5:
        mode = modes[i]
        if mode in ["r", "r+"]:
            with pytest.raises(IOError):
                a = ia.random.normal(shape, mu, sigma, cfg=cfg, dtype=dtype, mode=mode)
            mode = "a"
        a = ia.random.normal(shape, mu, sigma, cfg=cfg, dtype=dtype, mode=mode)
        b = np.random.normal(mu, sigma, size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False


# Bernoulli (compare against np.random.binomial)
@pytest.mark.parametrize(
    "p",
    [0.01, 0.15, 0.6],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data2,
)
def test_bernoulli(p, shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))

    modes = ["r", "r+", "a", "w", "w-"]
    i = 0
    while i < 5:
        mode = modes[i]
        if mode in ["r", "r+"]:
            with pytest.raises(IOError):
                a = ia.random.bernoulli(shape, p, cfg=cfg, dtype=dtype, mode=mode)
            mode = "w"
        a = ia.random.bernoulli(shape, p, cfg=cfg, dtype=dtype, mode=mode)
        b = np.random.binomial(1, p, size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False


# Binomial
@pytest.mark.parametrize(
    "n, p",
    [
        (10, 0.01),
        (5, 0.6),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data2,
)
def test_binomial(n, p, shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))

    modes = ["r", "r+", "a", "w", "w-"]
    i = 0
    while i < 5:
        mode = modes[i]
        if mode in ["r", "r+"]:
            with pytest.raises(IOError):
                a = ia.random.binomial(
                    shape, n, p, cfg=cfg, random_gen=ia.RandomGen.MRG32K3A, dtype=dtype, mode=mode
                )
            mode = "a"
        a = ia.random.binomial(
            shape, n, p, cfg=cfg, random_gen=ia.RandomGen.MRG32K3A, dtype=dtype, mode=mode
        )
        b = np.random.binomial(n, p, size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False


# Poisson
@pytest.mark.parametrize(
    "lamb",
    [5, 0.15],
)
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    rand_data,
)
@pytest.mark.parametrize(
    "dtype",
    dtype_data2,
)
def test_poisson(lamb, shape, chunks, blocks, dtype):
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    size = int(np.prod(shape))

    modes = ["r", "r+", "a", "w", "w-"]
    i = 0
    while i < 5:
        mode = modes[i]
        if mode in ["r", "r+"]:
            with pytest.raises(IOError):
                a = ia.random.poisson(
                    shape, lamb, cfg=cfg, random_gen=ia.RandomGen.MRG32K3A, dtype=dtype, mode=mode
                )
            mode = "w"
        a = ia.random.poisson(
            shape, lamb, cfg=cfg, random_gen=ia.RandomGen.MRG32K3A, dtype=dtype, mode=mode
        )
        b = np.random.poisson(lamb, size).reshape(shape).astype(dtype)
        c = ia.numpy2iarray(b, cfg=cfg)

        if not ia.random.kstest(a, c):
            i += 1
        else:
            return
    assert False
