import iarray as ia
import numpy as np

# Number of iterations per benchmark
NITER = 10

# Define array params
shape = [100000]
pshape = [10000]
bshape = [1000]
nthreads = 4
dtype = np.float64

storage = ia.StorageProperties(backend="blosc", chunkshape=pshape, blockshape=bshape,
                               enforce_frame=False, filename=None)

# Create initial containers
a1 = ia.linspace(ia.dtshape(shape, dtype), .01, .2, storage=storage)
a2 = np.linspace(.01, .2, shape[0], dtype=dtype).reshape(shape)


print("iarray evaluation...")

# And now, the expression
eval_flags = ia.EvalFlags(method="iterblosc2", engine="auto")
expr = ia.Expr(eval_flags=eval_flags, nthreads=nthreads)
expr.bind("x", a1)
expr.bind_out_properties(ia.dtshape(shape, np.float64), storage=storage)
expr.compile("(x - 1.35) * (x - 4.45) * (x - 8.5)")
b1 = expr.eval()
b1_n = ia.iarray2numpy(b1)

print("numpy evaluation...")
b2 = eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a2})

try:
    np.testing.assert_almost_equal(b2, b1_n)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
