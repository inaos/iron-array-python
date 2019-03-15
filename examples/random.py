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

a = np.random.poisson(5, size)


plt.hist(a, bins='auto', density=True, histtype="step", cumulative=False)

plt.show()
