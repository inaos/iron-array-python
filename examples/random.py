import iarray as ia
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

cfg = ia.Config()
ctx = ia.Context(cfg)
r_ctx = ia.RandomContext(ctx)

size = 1000000
shape = [size]

beta = 2

a = ia.random_exponential(ctx, r_ctx, beta, shape)

b = ia.iarray2numpy(ctx, a)

c = np.random.exponential(beta, size)


plt.hist(b, bins='auto', density=True, histtype="step")
plt.hist(c, bins='auto', density=True, histtype="step")
plt.show()
