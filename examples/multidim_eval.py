# Comparing performance of iarray vs numpy and numexpr (transcendental expressions and multidim arrays).
# You will need to install numexpr for this.

from time import time
import iarray as ia
import numpy as np
import numexpr as ne


# Define array params
dtype = np.float64
shape = [16000, 8000]
dtshape = ia.DTShape(shape, dtype)
nthreads = 8  # maximum number of threads to use

sexpr = "(cos(x) - sin(y)) * (x - 1.35) * (y - 4.45)"

# Create initial arrays.  You may opt to use automatic chunks and blocks,
# but you typically get optimal results when you fine-tune them.
store = ia.Store(chunks=[1000, 800], blocks=[100, 100])
ia.set_config(store=store, fp_mantissa_bits=24, nthreads=nthreads)

size = shape[0] * shape[1]
np0 = np.linspace(0, 10, size, dtype=dtype).reshape(shape)
ia0 = ia.numpy2iarray(np0)
np1 = np.linspace(0, 1, size, dtype=dtype).reshape(shape)
ia1 = ia.numpy2iarray(np1)

t0 = time()
np2 = eval("(np.cos(np0) - np.sin(np1)) * (np0 - 1.35) * (np1 - 4.45)")
t1 = time()
print("Time for numpy evaluation: %.3f" % (t1 - t0))

ne.set_num_threads(nthreads)
t0 = time()
np3 = ne.evaluate(sexpr, {"x": np0, "y": np1})
t1 = time()
print("Time for numexpr evaluation: %.3f" % (t1 - t0))

try:
    np.testing.assert_almost_equal(np3, np2)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")

t0 = time()
expr = ia.expr_from_string(sexpr, {"x": ia0, "y": ia1})

ia2 = expr.eval()
t1 = time()
print("Time for iarray evaluation: %.3f (cratio: %.2fx)" % ((t1 - t0), ia2.cratio))
np3 = ia.iarray2numpy(ia2)

try:
    np.testing.assert_almost_equal(np3, np2, decimal=5)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
