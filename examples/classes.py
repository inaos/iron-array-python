import iarray as ia
import numpy as np

shape = [20 * 10]
pshape = [10]
size = int(np.prod(shape))

cfg = ia.Config(eval_flags="iterblock", blocksize=16 * 1024)
ctx = ia.Context(cfg)

x = ia.linspace(ctx, size, 0, 10, shape=shape, pshape=pshape)

e = ia.Expression(ctx)

e.bind(b'x', x)

e.compile(b'x + x + 1')

z = e.eval(shape, pshape, dtype="double")

a = ia.iarray2numpy(ctx, z)

print(a)

