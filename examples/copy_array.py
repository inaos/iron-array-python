# Exercises the copy() with actual arrays and views of them too.

from time import time
import iarray as ia
import numpy as np
from dataclasses import asdict

# Create an ironArray array
a = ia.linspace([10000, 10000], -10, 10, chunks=[2000, 2000], blocks=[256, 256], dtype=np.float64)
print(f"src chunks: {a.chunks}")

# Do a regular copy changing the parameters
t0 = time()
with ia.config(chunks=[2048, 2048], blocks=[256, 256], codec=ia.Codecs.LZ4, clevel=5):
    b = a.copy()
t1 = time()

print(f"Time to make a copy with with (cont -> cont): {t1 - t0:.5f}")
print(f"- Chunks: {b.chunks}")


# Do a regular copy
t0 = time()
c = a.copy()
t1 = time()

print(f"Time to make a copy (cont -> cont): {t1 - t0:.5f}")
print(f"- Chunks: {c.chunks}")

# Get a slice (view)
v = a[100:1100, 200:1200]
t0 = time()
d = v.copy()
t1 = time()


print(f"Time to make a copy (view -> cont): {t1 - t0:.5f}")
print(f"- View chunks: {v.chunks}")
print(f"- Chunks: {d.chunks}")
