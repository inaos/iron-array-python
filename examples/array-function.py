import iarray as ia
import numpy as np
from time import time


# Define array params
shape = [10000, 2000]
pshape = [1000, 200]
nthreads = 4
eval_method = "iterblock"
kwargs = dict(eval_flags=eval_method, nthreads=nthreads, clevel=1, clib=ia.LZ4)

# Create initial containers
ia1 = ia.linspace(ia.dtshape(shape, pshape, np.float64), 0, 10, **kwargs)
np1 = ia.iarray2numpy(ia1)

t0 = time()
expr = ia.Expr(**kwargs)
expr.bind("x", ia1)
expr.compile("cos(x)")
ia2 = expr.eval(shape, pshape, np.float64)
t1 = time()
print("Time for iarray evaluation: %.3f (cratio: %.2fx)" % ((t1 - t0), ia2.cratio))
np2 = ia.iarray2numpy(ia2)

t0 = time()
np3 = np.cos(np1)
t1 = time()
print("Time for numpy evaluation: %.3f" % (t1 - t0))

try:
    np.testing.assert_almost_equal(np3, np2)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")

t0 = time()
ia3 = np.cos(ia1).eval(pshape=pshape, **kwargs)
# ia3 = ia1.cos().eval(pshape=pshape, **kwargs)
# ia3 = ia1.cos().eval(method="numexpr", pshape=pshape, **kwargs)
t1 = time()
print("iarray evaluation via __array_ufunc__: %.3f (cratio: %.2fx)" % ((t1 - t0), ia3.cratio))
np4 = ia.iarray2numpy(ia3)

try:
    np.testing.assert_almost_equal(np4, np2)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
