# This uses different computation methods inside (and outside) iarray and compares performance.

from time import time
import iarray as ia
import numpy as np
import numexpr as ne


# Define array params
dtype = np.float64
shape = [10000, 8000]
cshape = [1000, 800]
bshape = [100, 100]
nthreads = 8
eval_method = ia.EVAL_ITERBLOSC
storage = ia.StorageProperties(backend="blosc", chunkshape=cshape, blockshape=bshape,
                               enforce_frame=False, filename=None)
kwargs = dict(eval_method=eval_method, storage=storage, nthreads=nthreads, clevel=9, clib=ia.LZ4)


# Create initial arrays
ia1 = ia.linspace(ia.dtshape(shape, dtype), 0, 10, **kwargs)
np1 = ia.iarray2numpy(ia1)

t0 = time()
expr = ia.Expr(**kwargs)
expr.bind("x", ia1)
expr.bind_out_properties(ia.dtshape(shape, dtype), storage=storage)
expr.compile("cos(x)")
ia2 = expr.eval()
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

ne.set_num_threads(nthreads)
t0 = time()
np3 = ne.evaluate("cos(np1)")
t1 = time()
print("Time for numexpr evaluation: %.3f" % (t1 - t0))

try:
    np.testing.assert_almost_equal(np3, np2)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")

t0 = time()
parser = ia.Parser()
np3 = parser.evaluate("cos(np1)", {"np1": np1})
t1 = time()
print("Time for internal expression_eval evaluation: %.3f" % (t1 - t0))

try:
    np.testing.assert_almost_equal(np3, np2)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")

t0 = time()
ia3 = ia1.cos().eval(**kwargs)
# ia3 = ia1.cos().eval(method="numexpr", **kwargs)
t1 = time()
print("iarray evaluation via __array_ufunc__: %.3f (cratio: %.2fx)" % ((t1 - t0), ia3.cratio))
np4 = ia.iarray2numpy(ia3)

try:
    np.testing.assert_almost_equal(np4, np2)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
