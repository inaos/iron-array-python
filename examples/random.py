import iarray as ia
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

cfg = ia.Config()
ctx = ia.Context(cfg)
r_ctx = ia.RandomContext(ctx)

size = 1000000
shape = [size]

mu = 1
sigma = 0.5

a = ia.random_lognormal(ctx, r_ctx, mu, sigma, shape)

b = ia.iarray2numpy(ctx, a)

c = np.random.lognormal(mu, sigma, size)


plt.hist(b, bins='auto', density=True, histtype="step")
plt.hist(c, bins='auto', density=True, histtype="step")
plt.show()
