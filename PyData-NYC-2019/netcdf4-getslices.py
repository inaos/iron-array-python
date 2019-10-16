from time import time
import numpy as np
import netCDF4


NSLICES = 50
SLICE_THICKNESS = 10
IN_MEMORY = False

t0 = time()
if IN_MEMORY:
    with open("ia-data/rea6/tot_prec/2018.nc", "rb") as f:
        data = f.read()
    rootgrp = netCDF4.Dataset("memory", mode='r', memory=data)
else:
    rootgrp = netCDF4.Dataset("ia-data/rea6/tot_prec/2018.nc", mode='r')
precipitation = rootgrp['precipitation']
t1 = time()
print("Time to open file: %.3f" % (t1 - t0))
print("dataset:", precipitation)

# Get a random number of slices
nt, nx, ny = precipitation.shape
tslices = np.random.choice(nt - 1, NSLICES)

t0 = time()
sl = []
for tslice in tslices:
    sl.append(precipitation[slice(tslice, tslice + SLICE_THICKNESS),:,:])
t1 = time()
print("Time for getting %d slices: %.3f" % (NSLICES, (t1 - t0)))

# Time for getting the accumulation
t0 = time()
slsum1 = []
for i in range(NSLICES):
    slsum1.append(sl[i].sum())
t1 = time()
print("Time for summing up %d slices (via numpy): %.3f" % (NSLICES, (t1 - t0)))
