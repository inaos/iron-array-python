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
ia.random.binomial(shape, 5, 0.3)
t1 = time()
print(f"binomial: {t1 - t0:.2f}s")

t0 = time()
ia.linspace(shape, 0.0, 1.0)
t1 = time()
print(f"linspace: {t1 - t0:.2f}s")

t0 = time()
ia.arange(shape, 0.0, np.prod(shape) / 1)
t1 = time()
print(f"arange: {t1 - t0:.2f}s")
