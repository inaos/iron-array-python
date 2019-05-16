import iarray as ia
import numpy as np
import numexpr as ne


# Create iarray context
cfg = ia.Config(eval_flags="iterblock", blocksize=0)
ctx = ia.Context(cfg)

# Define array params
shape = [10000, 2000]
pshape = [1000, 200]
size = int(np.prod(shape))

# Create initial containers
a1 = ia.linspace(ctx, size, 0, 10, shape, pshape, "double")
a2 = ia.iarray2numpy(ctx, a1)

# Create iarray expression
print(f"Evaluación en iarray...")
expr = ia.Expression(ctx)
expr.bind("x", a1)
expr.compile("(x - 1.35) * (x - 4.45) * (x - 8.5)")
b2 = expr.eval(shape, pshape, "double")

b2_n = ia.iarray2numpy(ctx, b2)

# Numexpr
print(f"Evaluación en numexpr...")
b2 = ne.evaluate("(x - 1.35) * (x - 4.45) * (x - 8.5)", local_dict={"x": a2})

try:
    np.testing.assert_almost_equal(b2, b2_n)
    print(f"Los resultados son iguales. OK.")
except Exception:
    print(f"Los resultados no son iguales. ERROR.")
