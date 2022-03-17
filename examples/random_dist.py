# Example for creating a random distribution array and store it on a file

import iarray as ia
import numpy as np
import matplotlib.pyplot as plt


shape = [12, 31, 11, 22]
chunks = [4, 3, 5, 2]
blocks = [2, 2, 2, 2]

cfg = ia.Config(chunks=chunks, blocks=blocks)
size = int(np.prod(shape))

a = ia.random.random_sample(shape, cfg=cfg, dtype=np.float32)
b = np.random.rand(size).reshape(shape)
c = ia.numpy2iarray(b, cfg=cfg)

print(ia.random.kstest(a, c))
