import iarray as ia
import numpy as np

ia.init()

cfg = ia.config_new()

ctx = ia.context_new(cfg)

shape = (4, 8)
pshape = (2, 3)

size = np.prod(shape, dtype=np.int64)

dtshape = ia.dtshape_new(shape, pshape)

a = ia.arange(ctx, dtshape, 0, size, 1)

b = ia.iarray2numpy(ctx, a)

print(b)

c = np.linspace(0, 0.99, 100, dtype=np.float64).reshape(10, 10)

d = ia.numpy2iarray(ctx, c, pshape, b'linspace.iarray')
ia.container_free(ctx, d)

e = ia.from_file(ctx, b'linspace.iarray')

f = ia.iarray2numpy(ctx, e)

print(f)

ia.container_free(ctx, a)
ia.container_free(ctx, e)
ia.context_free(ctx)

ia.destroy()
