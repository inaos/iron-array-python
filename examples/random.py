import iarray as ia
import numpy as np
import matplotlib.pyplot as plt

cfg = ia.Config()
ctx = ia.Context(cfg)
r_ctx = ia.RandomContext(ctx)

size = 1000000
shape = [size]

a = ia.random_randn(ctx, r_ctx, shape)

b = ia.iarray2numpy(ctx, a)
c = np.random.randn(size)

plt.hist(b, bins='auto')
plt.hist(c, bins='auto')

plt.show()
