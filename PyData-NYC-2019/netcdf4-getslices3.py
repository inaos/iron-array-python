import io
import os
from time import time
import numpy as np
import netCDF4
import dask
import dask.array as da
import h5py


NSLICES = 50
SLICE_THICKNESS = 10
NTHREADS = 4

MEMPROF = True
if MEMPROF:
    from memory_profiler import profile
else:
    def profile(f):
        return f


rootgrp = None
@profile
def open_datafile(filename):
    global rootgrp
    rootgrp = netCDF4.Dataset(filename, mode='r')
    return rootgrp['precipitation']
t0 = time()
precipitation = open_datafile("ia-data/rea6/tot_prec/2018.nc")
t1 = time()
print("Time to open file: %.3f" % (t1 - t0))
print("dataset:", precipitation)

# Get a random number of slices
nt, nx, ny = precipitation.shape
shape = (NSLICES * SLICE_THICKNESS, nx, ny)
pshape = (1, nx, ny)
# tslices = np.random.choice(nt - SLICE_THICKNESS, NSLICES)
# np.random.seed(1)
np.random.seed(3)
tslice = np.random.choice(nt - NSLICES * SLICE_THICKNESS)

iobytes = io.BytesIO()
@profile
def concatenate_slices(dataset):
    # HDF5 is handier for outputing datasets
    f = h5py.File(iobytes)
    data = f.create_dataset("prec2", shape, chunks=pshape, compression="gzip", compression_opts=1, shuffle=True)
    for i in range(NSLICES * SLICE_THICKNESS):
        data[i] = dataset[i]
    return data
t0 = time()
prec2 = concatenate_slices(precipitation)
t1 = time()
print("Time for concatenating %d slices into HDF5 container: %.3f" % (NSLICES, (t1 - t0)))
print("cratio", NSLICES * SLICE_THICKNESS * nx * ny * 4 / len(iobytes.getvalue()))

# Compute the accumulation of the random slices into one
@profile
def compute_expr(x):
    scheduler = "single-threaded" if NTHREADS == 1 else "threads"
    with dask.config.set(scheduler=scheduler):
        dx = da.from_array(x)
        res = (np.sin(dx) - 3.2) * (np.cos(dx) + 1.2)
        # I don't see a way to use a memory handler for output
        da.to_hdf5("outarray.h5", "/prec2_computed", res)
        with h5py.File("outarray.h5") as f:
            data = f["/prec2_computed"]
    return data
t0 = time()
b2 = compute_expr(prec2)
t1 = time()
sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"
print("Time for computing '%s' expression (via dask + HDF5): %.3f" % (sexpr, (t1 - t0)))

@profile
def sum_concat(data):
    concatsum = 0
    for i in range(len(data)):
        concatsum += data[i].sum()
    return concatsum
t0 = time()
concatsum = sum_concat(prec2)
t1 = time()
print("Time for summing up the concatenated HDF5 container: %.3f" % (t1 - t0))

