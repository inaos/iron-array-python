import iarray as ia
import numpy as np
import numexpr as ne


# Create iarray context
cfg = ia.Config(eval_flags="iterblock", blocksize=0)
ctx = ia.Context(cfg)

# Define array params
shape = [100 * 200 * 1000]
pshape = [200 * 1000]
size = int(np.prod(shape))

# Create initial containers
a1 = ia.linspace(ctx, size, 0, 10, shape, pshape, "double")
a2 = ia.iarray2numpy(ctx, a1)

# Create iarray expression
expr = ia.Expression(ctx)
expr.bind(b'x', a1)
expr.compile(b'(x - 1.35) * (x - 4.45) * (x - 8.5)')
b2 = expr.eval(shape, pshape, "double")

b2_n = ia.iarray2numpy(ctx, b2)
print(b2_n)

# Numexpr
b2 = ne.evaluate("(x - 1.35) * (x - 4.45) * (x - 8.5)", local_dict={"x": a2})
print(b2)

