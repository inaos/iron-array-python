import iarray as ia
import numpy as np
import numexpr as ne

# Init iarray
ia.init()

# Create iarray context
cfg = ia.config_new(eval_flags="iterblock", blocksize=0)
ctx = ia.context_new(cfg)

# Define array params
shape = [100 * 200 * 1000]
pshape = [200 * 1000]
size = int(np.prod(shape))

# Create iarray containers
a = ia.linspace(ctx, size, 0, 10)
x = ia.iarray2numpy(ctx, a)

b = ia.container_new(ctx, shape)

# Create iarray expression
expr = ia.expr_new(ctx)

# Bind iarray containers
ia.expr_bind(expr, b'x', a)

# Compile a+b expression
ia.expr_compile(expr, b'(x - 1.35) * (x - 4.45) * (x - 8.5)')

# Eval expression
ia.expr_eval(expr, b)
c = ia.iarray2numpy(ctx, b)
print(c)

# Numexpr
d = ne.evaluate("(x - 1.35) * (x - 4.45) * (x - 8.5)")
print(d)

# Free data
ia.expr_free(ctx, expr)
ia.container_free(ctx, a)
ia.container_free(ctx, b)
ia.context_free(ctx)

# Destroy iarray
ia.destroy()

