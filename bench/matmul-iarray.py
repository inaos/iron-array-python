import iarray as ia
import numpy as np
import sys
import os
from time import time

persistent = True

# shape = (16384, 16384)
# chunkshape = (4096, 4096)
# blockshape = (1024, 1024)

shape = (1000, 1000)
chunkshape = (500, 500)
blockshape = (100, 100)

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

dtshape = ia.DTShape(shape, np.float64)

astorage = ia.Store(chunkshape, blockshape, urlpath=aurlpath)
bstorage = ia.Store(chunkshape, blockshape, urlpath=burlpath)
cstorage = ia.Store(chunkshape, blockshape, urlpath=curlpath)

ia.set_config(codec=codec, clevel=clevel, nthreads=nthreads)
cfg = ia.get_config()

print("(Re-)Generating operand A")
if persistent:
    if not os.path.exists(aurlpath):
        aia = ia.linspace(dtshape, -1, 1, storage=astorage)
    else:
        aia = ia.open("a.iarray")
        if (
            aia.dtshape.shape != shape
            or aia.chunkshape != chunkshape
            or aia.blockshape != blockshape
        ):
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            os.remove(aurlpath)
            aia = ia.linspace(dtshape, -1, 1, storage=astorage)
else:
    aia = ia.linspace(dtshape, -1, 1, storage=astorage)

print("(Re-)Generating operand B")
if persistent:
    if not os.path.exists(burlpath):
        bia = ia.linspace(dtshape, -1, 1, storage=bstorage)
    else:
        bia = ia.open("b.iarray")
        if (
            bia.dtshape.shape != shape
            or bia.chunkshape != chunkshape
            or bia.blockshape != blockshape
        ):
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            os.remove(burlpath)
            bia = ia.linspace(dtshape, -1, 1, storage=bstorage)
else:
    bia = ia.linspace(dtshape, -1, 1, storage=bstorage)

if persistent:
    if os.path.exists(curlpath):
        os.remove(curlpath)

print(f"Start actual matmul with nthreads = {cfg.nthreads}")
t0 = time()
cia = ia.matmul(aia, bia, storage=cstorage)
print("Time for iarray matmul:", round((time() - t0), 3))

if persistent:
    # if os.path.exists(aurlpath):
    #     os.remove(aurlpath)
    # if os.path.exists(burlpath):
    #     os.remove(burlpath)
    if os.path.exists(curlpath):
        os.remove(curlpath)
