
# Matrix matrix multiplication performance


```python
import iarray as ia
import numpy as np
from itertools import zip_longest as izip
from time import time
import ctypes


mkl_rt = ctypes.CDLL('libmkl_rt.dylib')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
nrep = 10
```

## Sequencial


```python
mkl_set_num_threads(1)
cfg = ia.Config(max_num_threads=1, compression_level=0)
ctx = ia.Context(cfg)
```

### Plainbuffer


```python
shape = [2000, 2000]
pshape = None
bshape = [2000, 2000]
size = int(np.prod(shape))

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0)/nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0)/nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.4656 s
    Time to compute matmul with iarray: 0.4606 s
    Numpy is 0.9893x more faster than iarray


### Superchunk


```python
shape = [2000, 2000]
pshape = [200, 200]
bshape = [200, 200]
size = int(np.prod(shape))

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0)/nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0)/nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.4618 s
    Time to compute matmul with iarray: 0.6456 s
    Numpy is 1.3979x more faster than iarray


## Multithreading


```python
mkl_set_num_threads(2)
cfg = ia.Config(max_num_threads=2, compression_level=0)
ctx = ia.Context(cfg)
```

### Plainbuffer


```python
shape = [2000, 2000]
pshape = None
bshape = [2000, 2000]
size = int(np.prod(shape))

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0)/nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0)/nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.2224 s
    Time to compute matmul with iarray: 0.2407 s
    Numpy is 1.082x more faster than iarray


### Superchunk


```python
shape = [2000, 2000]
pshape = [200, 200]
bshape = [200, 200]
size = int(np.prod(shape))

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0)/nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0)/nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.2241 s
    Time to compute matmul with iarray: 0.4427 s
    Numpy is 1.9754x more faster than iarray

