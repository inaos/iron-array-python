import iarray as ia
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

cfg = ia.Config()
ctx = ia.Context(cfg)
r_ctx = ia.RandomContext(ctx)

size = 10000
shape = [size]
pshape = [100]

a = np.random.normal(3, 0.5, size).astype(np.float32)
c = ia.numpy2iarray(ctx, a, pshape=pshape, filename="test_normal_f_3_05.iarray")

b = ia.random_uniform(ctx, r_ctx, 0.3, 0.5, shape, pshape)
d = ia.iarray2numpy(ctx, b)

plt.hist(a, bins='auto', density=True, histtype="step")
plt.hist(d, bins='auto', density=True, histtype="step")

plt.show()
