# This uses different computation methods inside (and outside) iarray and compares performance.

from time import time
import iarray as ia
import numpy as np
import numexpr as ne


# Define array params
dtype = np.float64
shape = [1000, 8000]
cshape = [1000, 800]
bshape = [100, 100]
ia.set_config(chunkshape=cshape, blockshape=bshape)
dtshape = ia.DTShape(shape, dtype)

# Create initial arrays
ia1 = ia.linspace(dtshape, 0, 10)
#ia1 = ia.zeros(dtshape)  # exposes a flaw in expr.eval()
np1 = ia.iarray2numpy(ia1)

t0 = time()
np2 = np.cos(np1)
t1 = time()
print("Time for numpy evaluation: %.3fs" % (t1 - t0))

t0 = time()
np3_ = ne.evaluate("cos(np1)")
t1 = time()
print("Time for numexpr evaluation: %.3fs" % (t1 - t0))

t0 = time()
expr = ia.expr_from_string("cos(x)", {"x": ia1})
ia2 = expr.eval()
t1 = time()
print("Time for iarray evaluation: %.3fs (cratio: %.2fx)" % ((t1 - t0), ia2.cratio))
np_ = ia.iarray2numpy(ia2)
ia.cmp_arrays(np_, np2, "OK.  Results are the same.")

t0 = time()
expr = ia.expr_from_string("cos(x)", {"x": ia1})
expr.register_as_postfilter(ia1)
ia3 = ia1.copy()
t1 = time()
print("Time for iarray postfilters (via copy): %.3fs (cratio: %.2fx)" % ((t1 - t0), ia3.cratio))
np_ = ia.iarray2numpy(ia3)
ia.cmp_arrays(np_, np2, "OK.  Results are the same.")

t0 = time()
expr = ia.expr_from_string("cos(x)", {"x": ia1})
expr.register_as_postfilter(ia1)
np_ = ia1.data
t1 = time()
print("Time for iarray postfilters: %.3fs (via .data)" % (t1 - t0))
ia.cmp_arrays(np_, np2, "OK.  Results are the same.")
