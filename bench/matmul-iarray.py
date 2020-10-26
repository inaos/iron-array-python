import iarray as ia
import numpy as np
import sys
import os

persistent = True

shape = (16384, 16384)
chunkshape = (4096, 4096)
blockshape = (1024, 1024)

shape = (1000, 1000)
chunkshape = (500, 500)
blockshape = (250, 250)

if len(sys.argv) == 1:
    nthreads = 1
else:
    nthreads = int(sys.argv[1])

clib = ia.LZ4
clevel = 5

if persistent:
    afilename = "a.iarray"
    bfilename = "b.iarray"
    cfilename = "c.iarray"
else:
    afilename = None
    bfilename = None
    cfilename = None

dtshape = ia.dtshape(shape, np.float64)

astorage = ia.StorageProperties(chunkshape, blockshape, filename=afilename)
bstorage = ia.StorageProperties(chunkshape, blockshape, filename=bfilename)
cstorage = ia.StorageProperties(chunkshape, blockshape, filename=cfilename)

cparams = dict(clib=clib, clevel=clevel, nthreads=nthreads)

if persistent:
    if not os.path.exists(afilename):
        aia = ia.linspace(dtshape, -1, 1, storage=astorage, **cparams)
    else:
        aia = ia.load("a.iarray", load_in_mem=False)
else:
    aia = ia.linspace(dtshape, -1, 1, storage=astorage, **cparams)

if persistent:
    if not os.path.exists(bfilename):
        bia = ia.linspace(dtshape, -1, 1, storage=bstorage, **cparams)
    else:
        bia = ia.load("b.iarray", load_in_mem=False)
else:
    bia = ia.linspace(dtshape, -1, 1, storage=bstorage, **cparams)

print("start matmul")
cia = ia.matmul(aia, bia, storage=cstorage, **cparams)

if persistent:
    # if os.path.exists(afilename):
    #     os.remove(afilename)
    # if os.path.exists(bfilename):
    #     os.remove(bfilename)
    if os.path.exists(cfilename):
        os.remove(cfilename)
