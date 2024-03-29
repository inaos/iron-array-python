# This uses different computation methods inside (and outside) iarray and compares performance.

from time import time
import iarray as ia
import numpy as np
import numexpr as ne


# Define array params
dtype = np.float64
shape = [10000, 8000]
chunks = [1000, 800]
blocks = [100, 100]
ia.set_config_defaults(chunks=chunks, blocks=blocks, dtype=dtype)

# Create initial arrays
ia1 = ia.linspace(0, 10, num=int(np.prod(shape)), shape=shape)
np1 = ia.iarray2numpy(ia1)

t0 = time()
np2 = np.cos(np1)
t1 = time()
print("Time for numpy evaluation: %.3f" % (t1 - t0))

t0 = time()
np3_ = ne.evaluate("cos(np1)")
t1 = time()
print("Time for numexpr evaluation: %.3f" % (t1 - t0))

t0 = time()
expr = ia.expr_from_string("cos(x)", {"x": ia1}, favor=ia.Favor.SPEED)
ia2 = expr.eval()
t1 = time()
print("Time for iarray evaluation: %.3f (cratio: %.2fx)" % ((t1 - t0), ia2.cratio))

np3 = ia.iarray2numpy(ia2)
ia.cmp_arrays(np3, np2, "OK.  Results are the same.")

t0 = time()
ia3 = ia.cos(ia1).eval(favor=ia.Favor.SPEED)
t1 = time()
print("Time for iarray via lazy evaluation: %.3f (cratio: %.2fx)" % ((t1 - t0), ia3.cratio))

np4 = ia.iarray2numpy(ia3)
ia.cmp_arrays(np4, np2, "OK.  Results are the same.")
