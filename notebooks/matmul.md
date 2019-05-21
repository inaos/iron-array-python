
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

## Plainbuffer


```python
shape = [2000, 2000]
pshape = None
bshape = [2000, 2000]
size = int(np.prod(shape))
```

### Sequential


```python
mkl_set_num_threads(1)
cfg = ia.Config(max_num_threads=1)
ctx = ia.Context(cfg)

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0) / nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0) / nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.4741 s
    Time to compute matmul with iarray: 0.5281 s
    Numpy is 1.114x more faster than iarray


### Multithreading


```python
mkl_set_num_threads(2)
cfg = ia.Config(max_num_threads=2)
ctx = ia.Context(cfg)

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0) / nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0) / nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.3497 s
    Time to compute matmul with iarray: 0.4093 s
    Numpy is 1.1702x more faster than iarray


## Superchunk (without compression)


```python
shape = [2000, 2000]
pshape = [200, 200]
bshape = [200, 200]
size = int(np.prod(shape))
```

### Sequential


```python
mkl_set_num_threads(1)
cfg = ia.Config(max_num_threads=1, compression_level=0)
ctx = ia.Context(cfg)

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0) / nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0) / nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.4719 s
    Time to compute matmul with iarray: 0.7908 s
    Numpy is 1.6758x more faster than iarray


### Multithreading


```python
mkl_set_num_threads(2)
cfg = ia.Config(max_num_threads=2, compression_level=0)
ctx = ia.Context(cfg)

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0) / nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0) / nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.3086 s
    Time to compute matmul with iarray: 0.6997 s
    Numpy is 2.2673x more faster than iarray


## Superchunk (with compression)


```python
shape = [2000, 2000]
pshape = [200, 200]
bshape = [200, 200]
size = int(np.prod(shape))
```

### Sequential


```python
mkl_set_num_threads(1)
cfg = ia.Config(max_num_threads=1, compression_level=5)
ctx = ia.Context(cfg)

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0) / nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0) / nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.5016 s
    Time to compute matmul with iarray: 1.0413 s
    Numpy is 2.076x more faster than iarray


### Multithreading


```python
mkl_set_num_threads(2)
cfg = ia.Config(max_num_threads=2, compression_level=5)
ctx = ia.Context(cfg)

a = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.linspace(ctx, size, -1, 1, shape=shape, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()
t_np = (t1 - t0) / nrep

print(f"Time to compute matmul with numpy: {round(t_np, 4)} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, bshape, bshape)
t1 = time()
t_ia = (t1 - t0) / nrep

print(f"Time to compute matmul with iarray: {round(t_ia, 4)} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print(f"Numpy is {round(t_ia/t_np, 4)}x more faster than iarray")
```

    Time to compute matmul with numpy: 0.4197 s
    Time to compute matmul with iarray: 0.856 s
    Numpy is 2.0394x more faster than iarray

