# Comparing performance of iarray vs numpy and numexpr (transcendental expressions and multidim arrays).
# You will need to install numexpr for this.

from time import time
import iarray as ia
import numpy as np
import numexpr as ne


# Define array params
dtype = np.float64
shape = [16000, 8000]
dtshape = ia.dtshape(shape, dtype)
nthreads = 8  # maximum number of threads to use

sexpr = "(cos(%s) - sin(%s)) * (%s - 1.35) * (%s - 4.45)"
npexpr = "(np.cos(%s) - np.sin(%s)) * (%s - 1.35) * (%s - 4.45)"

# Create initial arrays.  You may opt to use automatic chunkshape and blockshape,
# but you typically get optimal results when you fine-tune them.
storage = ia.StorageProperties(chunkshape=[1000, 800], blockshape=[100, 100])
kwargs = dict(storage=storage, fp_mantissa_bits=24)

size = shape[0] * shape[1]
np0 = np.linspace(0, 10, size, dtype=dtype).reshape(shape)
ia0 = ia.numpy2iarray(np0, **kwargs)
np1 = np.linspace(0, 1, size, dtype=dtype).reshape(shape)
ia1 = ia.numpy2iarray(np1, **kwargs)

t0 = time()
np2 = eval(npexpr % (("np0", "np1") * 2))
t1 = time()
print("Time for numpy evaluation: %.3f" % (t1 - t0))

ne.set_num_threads(nthreads)
t0 = time()
np3 = ne.evaluate(sexpr % (("np0", "np1") * 2))
t1 = time()
print("Time for numexpr evaluation: %.3f" % (t1 - t0))

try:
    np.testing.assert_almost_equal(np3, np2)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")

t0 = time()
expr = ia.create_expr(sexpr % (("x", "y") * 2), {"x": ia0, "y": ia1}, dtshape, **kwargs)

ia2 = expr.eval()
t1 = time()
print("Time for iarray evaluation: %.3f (cratio: %.2fx)" % ((t1 - t0), ia2.cratio))
np3 = ia.iarray2numpy(ia2)

try:
    np.testing.assert_almost_equal(np3, np2, decimal=5)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
