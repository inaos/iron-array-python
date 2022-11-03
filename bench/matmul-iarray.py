import iarray as ia
import numpy as np
import os
from time import time

persistent = True
shape = (10_000, 10_000)

if persistent:
    aurlpath = "a.iarr"
    burlpath = "b.iarr"
    curlpath = "c.iarr"
else:
    aurlpath = None
    burlpath = None
    curlpath = None

# Obtain optimal chunk and block shapes
mparams = ia.matmul_params(shape, shape)
amchunks, amblocks, bmchunks, bmblocks = mparams
print(mparams)
# ia.set_config_defaults(chunks=amchunks, blocks=amblocks, favor=ia.Favor.SPEED)
# For some reason, disabling BTune make better cratio and similar speed.
ia.set_config_defaults(chunks=amchunks, blocks=amblocks, btune=False)

print("(Re-)Generating operand A")
if persistent:
    if not os.path.exists(aurlpath):
        a = ia.linspace(-1, 1, int(np.prod(shape)), shape=shape, urlpath=aurlpath)
    else:
        a = ia.open("a.iarr")
else:
    a = ia.linspace(-1, 1, int(np.prod(shape)), shape=shape, urlpath=aurlpath)

print("(Re-)Generating operand B")
if persistent:
    if not os.path.exists(burlpath):
        b = ia.linspace(-1, 1, int(np.prod(shape)), shape=shape, urlpath=burlpath)
    else:
        b = ia.open("b.iarr")
else:
    b = ia.linspace(-1, 1, int(np.prod(shape)), shape=shape, urlpath=burlpath)

if persistent:
    if os.path.exists(curlpath):
        ia.remove_urlpath(curlpath)

print(f"chunks a: {a.chunks}, b: {b.chunks}")
print(f"cratio a: {a.cratio:.2f}, b: {b.cratio:.2f}")
cfg = ia.get_config_defaults()
print(f"Start actual matmul with nthreads = {cfg.nthreads}")
t0 = time()
c = ia.matmul(a, b, urlpath=curlpath)
print("Time for iarray matmul:", round((time() - t0), 3))
print(f"cratio c: {c.cratio:.2f}")

if persistent:
    if os.path.exists(aurlpath):
        ia.remove_urlpath(aurlpath)
    if os.path.exists(burlpath):
        ia.remove_urlpath(burlpath)
    if os.path.exists(curlpath):
        ia.remove_urlpath(curlpath)
