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
    afilename = "a.iarray"
    bfilename = "b.iarray"
    cfilename = "c.iarray"
else:
    afilename = None
    bfilename = None
    cfilename = None

dtshape = ia.DTShape(shape, np.float64)

astorage = ia.Storage(chunkshape, blockshape, filename=afilename)
bstorage = ia.Storage(chunkshape, blockshape, filename=bfilename)
cstorage = ia.Storage(chunkshape, blockshape, filename=cfilename)

ia.set_config(codec=codec, clevel=clevel, nthreads=nthreads)
cfg = ia.get_config()

print("(Re-)Generating operand A")
if persistent:
    if not os.path.exists(afilename):
        aia = ia.linspace(dtshape, -1, 1, storage=astorage)
    else:
        aia = ia.open("a.iarray")
        if (
            aia.dtshape.shape != shape
            or aia.chunkshape != chunkshape
            or aia.blockshape != blockshape
        ):
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            os.remove(afilename)
            aia = ia.linspace(dtshape, -1, 1, storage=astorage)
else:
    aia = ia.linspace(dtshape, -1, 1, storage=astorage)

print("(Re-)Generating operand B")
if persistent:
    if not os.path.exists(bfilename):
        bia = ia.linspace(dtshape, -1, 1, storage=bstorage)
    else:
        bia = ia.open("b.iarray")
        if (
            bia.dtshape.shape != shape
            or bia.chunkshape != chunkshape
            or bia.blockshape != blockshape
        ):
            # Ooops, we cannot use the array on-disk.  Regenerate it.
            os.remove(bfilename)
            bia = ia.linspace(dtshape, -1, 1, storage=bstorage)
else:
    bia = ia.linspace(dtshape, -1, 1, storage=bstorage)

if persistent:
    if os.path.exists(cfilename):
        os.remove(cfilename)

print(f"Start actual matmul with nthreads = {cfg.nthreads}")
t0 = time()
cia = ia.matmul(aia, bia, storage=cstorage)
print("Time for iarray matmul:", round((time() - t0), 3))

if persistent:
    # if os.path.exists(afilename):
    #     os.remove(afilename)
    # if os.path.exists(bfilename):
    #     os.remove(bfilename)
    if os.path.exists(cfilename):
        os.remove(cfilename)
