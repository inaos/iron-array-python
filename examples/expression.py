import iarray as ia
import numpy as np


# Define array params
shape = [10000, 2000]
pshape = [1000, 200]
size = int(np.prod(shape))

# Create initial containers
a1 = ia.linspace2(size, 0, 10, shape, pshape, "double")
a2 = ia.iarray2numpy2(a1)

print("iarray evaluation...")
# Create iarray context
cfg = ia.Config(eval_flags="iterblock", blocksize=0)
ctx = ia.Context(cfg)
# And now, the expression
expr = ia.Expression(ctx)
expr.bind("x", a1)
expr.compile("(x - 1.35) * (x - 4.45) * (x - 8.5)")
b2 = expr.eval(shape, pshape, "double")

b2_n = ia.iarray2numpy2(b2)

print("numpy evaluation...")
b2 = eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a2})

try:
    np.testing.assert_almost_equal(b2, b2_n)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
