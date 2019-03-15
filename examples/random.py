import iarray as ia
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

cfg = ia.Config()
ctx = ia.Context(cfg)
r_ctx = ia.RandomContext(ctx)

size = 1000000
shape = [size]
pshape = [100]

a1 = ia.random_beta(ctx, r_ctx, 2, 5, shape, pshape)
a2 = ia.iarray2numpy(ctx, a1)

b = np.random.beta(2, 5, size)

plt.hist([a2, b], bins='auto', density=True, histtype="step", cumulative=False)

plt.show()
