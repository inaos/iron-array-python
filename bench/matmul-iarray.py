import iarray as ia
import numpy as np
import sys
import os
from time import time

persistent = True

# shape = (16384, 16384)
# chunks = (4096, 4096)
# blocks = (1024, 1024)

shape = (1000, 1000)
chunks = (500, 500)
blocks = (100, 100)

if len(sys.argv) == 1:
    nthreads = 8
else:
    nthreads = int(sys.argv[1])

codec = ia.Codecs.LZ4
clevel = 5

if persistent:
    aurlpath = "a.iarray"
    burlpath = "b.iarray"
    curlpath = "c.iarray"
else:
    aurlpath = None
    burlpath = None
    curlpath = None

astore = ia.Store(chunks, blocks, urlpath=aurlpath)
bstore = ia.Store(chunks, blocks, urlpath=burlpath)
cstore = ia.Store(chunks, blocks, urlpath=curlpath)

ia.set_config(codec=codec, clevel=clevel, nthreads=nthreads, dtype=np.float64)
cfg = ia.get_config()

print("(Re-)Generating operand A")
if persistent:
    if not os.path.exists(aurlpath):
        aia = ia.linspace(shape, -1, 1, store=astore)
    else:
        aia = ia.open("a.iarray")
        if aia.shape != shape or aia.chunks != chunks or aia.blocks != blocks:
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            os.remove(aurlpath)
            aia = ia.linspace(shape, -1, 1, store=astore)
else:
    aia = ia.linspace(shape, -1, 1, store=astore)

print("(Re-)Generating operand B")
if persistent:
    if not os.path.exists(burlpath):
        bia = ia.linspace(shape, -1, 1, store=bstore)
    else:
        bia = ia.open("b.iarray")
        if bia.shape != shape or bia.chunks != chunks or bia.blocks != blocks:
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            os.remove(burlpath)
            bia = ia.linspace(shape, -1, 1, store=bstore)
else:
    bia = ia.linspace(shape, -1, 1, store=bstore)

if persistent:
    if os.path.exists(curlpath):
        os.remove(curlpath)

print(f"Start actual matmul with nthreads = {cfg.nthreads}")
t0 = time()
cia = ia.matmul(aia, bia, store=cstore)
print("Time for iarray matmul:", round((time() - t0), 3))

if persistent:
    # if os.path.exists(aurlpath):
    #     os.remove(aurlpath)
    # if os.path.exists(burlpath):
    #     os.remove(burlpath)
    if os.path.exists(curlpath):
        os.remove(curlpath)
