import iarray as ia
import numpy as np
...
# Init iarray
ia.init()

# Create iarray context
cfg = ia.config_new()
ctx = ia.context_new(cfg)

# Define array params
shape = [1000]
pshape = [10]
size = int(np.prod(shape))

# Create dtshape
dtshape = ia.dtshape_new(shape, pshape, "double")

# Create iarray containers
a = ia.arange(ctx, dtshape, 0, size, 1)
b = ia.arange(ctx, dtshape, 0, size, 1)
c = ia.container_new(ctx, dtshape)

# Create iarray expression
expr = ia.expr_new(ctx)

# Bind iarray containers
ia.expr_bind(expr, b'a', a)
ia.expr_bind(expr, b'b', b)

# Compile a+b expression
ia.expr_compile(expr, b'a+1')

# Eval expression
ia.expr_eval(expr, c)
d = ia.iarray2numpy(ctx, c)
print(d)

# Free data
ia.expr_free(ctx, expr)
ia.container_free(ctx, a)
ia.container_free(ctx, b)
ia.container_free(ctx, c)
ia.context_free(ctx)

# Destroy iarray
ia.destroy()
