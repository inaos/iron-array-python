import numpy as np

import iarray as ia
from time import time


shape = (800, 800, 800)
chunks = (200, 200, 200)
blocks = (20, 20, 20)

ia.set_config_defaults(chunks=chunks, blocks=blocks, nthreads=8)

t0 = time()
ia.random.random_sample(shape)
t1 = time()
print(f"random_sample: {t1 - t0:.2f}s")

t0 = time()
ia.random.uniform(shape, 0.0, 1.0)
t1 = time()
print(f"uniform: {t1 - t0:.2f}s")

t0 = time()
np.random.uniform(0.0, 1.0, np.prod(shape)).reshape(shape)
t1 = time()
print(f"uniform (np): {t1 - t0:.2f}s")

t0 = time()
ia.random.binomial(shape, 5, 0.3, dtype=ia.int32)
t1 = time()
print(f"binomial: {t1 - t0:.2f}s")

t0 = time()
ia.linspace(0.0, 1.0, int(np.prod(shape)), shape=shape)
t1 = time()
print(f"linspace: {t1 - t0:.2f}s")

t0 = time()
np.linspace(0.0, 1.0, np.prod(shape)).reshape(shape)
t1 = time()
print(f"linspace (np): {t1 - t0:.2f}s")

t0 = time()
ia.arange(0.0, np.prod(shape) / 1, shape=shape)
t1 = time()
print(f"arange: {t1 - t0:.2f}s")
