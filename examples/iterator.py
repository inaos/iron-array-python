import iarray as ia
import numpy as np
import matplotlib.pyplot as plt

cfg = ia.Config()
ctx = ia.Context(cfg)
rnd_ctx = ia.RandomContext(ctx, seed=123)
np.random.seed(456)

shape = [1000, 1000]
pshape = [100, 100]

size = int(np.prod(shape))

c1 = ia.empty(ctx, shape, pshape)

for (elem_ind, part) in c1.iter_write():
    print(elem_ind)
    part[:] = np.random.randn(part.size).reshape(part.shape).astype(np.float64)

c2 = ia.random_randn(ctx, rnd_ctx, shape, pshape)

c1_n = ia.iarray2numpy(ctx, c1)
c2_n = ia.iarray2numpy(ctx, c2)

plt.hist(c1_n.flatten(), bins='auto', density=True, histtype="step", cumulative=False)
plt.hist(c2_n.flatten(), bins='auto', density=True, histtype="step", cumulative=False)

plt.show()

