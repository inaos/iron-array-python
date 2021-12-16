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

codec = ia.Codec.LZ4
clevel = 5

if persistent:
    aurlpath = "a.iarr"
    burlpath = "b.iarr"
    curlpath = "c.iarr"
else:
    aurlpath = None
    burlpath = None
    curlpath = None

acfg = ia.Config(chunks=chunks, blocks=blocks, urlpath=aurlpath)
bcfg = ia.Config(chunks=chunks, blocks=blocks, urlpath=burlpath)
ccfg = ia.Config(chunks=chunks, blocks=blocks, urlpath=curlpath)

ia.set_config_defaults(codec=codec, clevel=clevel, nthreads=nthreads, dtype=np.float64, btune=False)
cfg = ia.get_config_defaults()

print("(Re-)Generating operand A")
if persistent:
    if not os.path.exists(aurlpath):
        aia = ia.linspace(shape, -1, 1, cfg=acfg)
    else:
        aia = ia.open("a.iarr")
        if aia.shape != shape or aia.chunks != chunks or aia.blocks != blocks:
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            ia.remove_urlpath(aurlpath)
            aia = ia.linspace(shape, -1, 1, cfg=acfg)
else:
    aia = ia.linspace(shape, -1, 1, cfg=acfg)

print("(Re-)Generating operand B")
if persistent:
    if not os.path.exists(burlpath):
        bia = ia.linspace(shape, -1, 1, cfg=bcfg)
    else:
        bia = ia.open("b.iarr")
        if bia.shape != shape or bia.chunks != chunks or bia.blocks != blocks:
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            ia.remove_urlpath(burlpath)
            bia = ia.linspace(shape, -1, 1, cfg=bcfg)
else:
    bia = ia.linspace(shape, -1, 1, cfg=bcfg)

if persistent:
    if os.path.exists(curlpath):
        ia.remove_urlpath(curlpath)

print(f"Start actual matmul with nthreads = {cfg.nthreads}")
t0 = time()
cia = ia.matmul(aia, bia, cfg=ccfg)
print("Time for iarray matmul:", round((time() - t0), 3))

if persistent:
    # if os.path.exists(aurlpath):
    #     ia.remove_urlpath(aurlpath)
    # if os.path.exists(burlpath):
    #     ia.remove_urlpath(burlpath)
    if os.path.exists(curlpath):
        ia.remove_urlpath(curlpath)
