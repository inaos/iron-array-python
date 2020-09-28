import numpy as np

import iarray as ia
from iarray import udf
from iarray.udf import float64, int64

# Vector and sizes and chunking
shape = [1000 * 1000]
chunkshape = [400 * 1000]
blockshape = [16 * 1000]

expression = '(x - 1.35) * (x - 4.45) * (x - 8.5)'
clevel = 5   # compression level
clib = ia.LZ4  # compression codec
nthreads = 8  # number of threads for the evaluation and/or compression

engine = "udf"  # can be "udf" or "juggernaut" (it always works with "juggernaut")
backend = "plainbuffer"

@udf.jit(verbose=0)
def poly_llvm(out: udf.Array(float64, 1), x: udf.Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return 0

if backend == "plinbuffer":
    storage = ia.StorageProperties(plainbuffer=True)
else:
    storage = ia.StorageProperties(chunkshape, blockshape)

print(f"Backend: {backend}, engine: {engine}")
cparams = dict(clib=clib, clevel=clevel, nthreads=nthreads)

xa = ia.linspace(ia.dtshape(shape=shape), 0., 10., storage=storage, **cparams)

if engine == "udf":
    expr = poly_llvm.create_expr([xa], ia.dtshape(shape), storage=storage, **cparams)
    ya = expr.eval()
else:
    x = xa
    ya = eval("((x - 1.35) * (x - 4.45) * (x - 8.5))", {"x": x})
    ya = ya.eval(storage=storage, **cparams)

print("iarray evaluation done!")

# Compute with numpy and test results
N = int(np.prod(shape))
x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)
y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)
y1 = ia.iarray2numpy(ya)
np.testing.assert_almost_equal(y0, y1)