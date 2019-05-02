import iarray as ia
import numpy as np
import matplotlib.pyplot as plt

cfg = ia.Config(compression_codec=0)
ctx = ia.Context(cfg)
r_ctx = ia.RandomContext(ctx)

size = 10000
shape = [size]
pshape = [100]

a1 = ia.random_uniform(ctx, r_ctx, 0, 1, shape, pshape)
a2 = ia.iarray2numpy(ctx, a1)

b1 = np.random.poisson(0.6, size).astype(np.float32)

b2 = ia.numpy2iarray(ctx, b1, pshape=pshape, filename="test_poisson_f_06.iarray")

print(ia.random_kstest(ctx, a1, b2))

plt.hist([a2, b1], bins='auto', density=True, histtype="step", cumulative=False)

plt.show()
