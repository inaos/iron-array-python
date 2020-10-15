import iarray as ia
import numpy as np
import sys

persistent = True

shape = (16384, 16384)
chunkshape = (4096, 4096)
blockshape = (1024, 1024)

if len(sys.argv) == 1:
    nthreads = 1
else:
    nthreads = int(sys.argv[1])

clib = ia.LZ4
clevel = 5

if persistent:
    import os

    afilename = "a.iarray"
    if os.path.exists(afilename):
        os.remove(afilename)
    bfilename = "b.iarray"
    if os.path.exists(bfilename):
        os.remove(bfilename)
    cfilename = "c.iarray"
    if os.path.exists(cfilename):
        os.remove(cfilename)
else:
    afilename = None
    bfilename = None
    cfilename = None

dtshape = ia.dtshape(shape, np.float64)

astorage = ia.StorageProperties(chunkshape, blockshape, filename=afilename)
bstorage = ia.StorageProperties(chunkshape, blockshape, filename=bfilename)
cstorage = ia.StorageProperties(chunkshape, blockshape, filename=cfilename)

cparams = dict(clib=clib, clevel=clevel, nthreads=nthreads)


@profile
def create_data():
    aia = ia.linspace(dtshape, -1, 1, storage=astorage, **cparams)
    bia = ia.linspace(dtshape, -1, 1, storage=bstorage, **cparams)
    return aia, bia


aia, bia = create_data()


@profile
def matmul():
    cia = ia.matmul(aia, bia, storage=cstorage, **cparams)
    return cia


cia = matmul()


if persistent:
    if os.path.exists(afilename):
        os.remove(afilename)
    if os.path.exists(bfilename):
        os.remove(bfilename)
    if os.path.exists(cfilename):
        os.remove(cfilename)

print(aia.cratio)
print(bia.cratio)
print(cia.cratio)
