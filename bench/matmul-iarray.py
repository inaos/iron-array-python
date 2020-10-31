import iarray as ia
import numpy as np
import sys
import os

persistent = True

# shape = (16384, 16384)
# chunkshape = (4096, 4096)
# blockshape = (1024, 1024)

shape = (500, 500)
chunkshape = (250, 250)
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

if persistent:
    if not os.path.exists(afilename):
        aia = ia.linspace(dtshape, -1, 1, storage=astorage)
    else:
        aia = ia.load("a.iarray", load_in_mem=False)
else:
    aia = ia.linspace(dtshape, -1, 1, storage=astorage)

if persistent:
    if not os.path.exists(bfilename):
        bia = ia.linspace(dtshape, -1, 1, storage=bstorage)
    else:
        bia = ia.load("b.iarray", load_in_mem=False)
else:
    bia = ia.linspace(dtshape, -1, 1, storage=bstorage)

print(f"start matmul with nthreads={nthreads}")
cia = ia.matmul(aia, bia, storage=cstorage)

if persistent:
    # if os.path.exists(afilename):
    #     os.remove(afilename)
    # if os.path.exists(bfilename):
    #     os.remove(bfilename)
    if os.path.exists(cfilename):
        os.remove(cfilename)
