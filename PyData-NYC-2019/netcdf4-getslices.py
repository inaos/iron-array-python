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
IN_MEMORY = False
NTHREADS = 4

MEMPROF = True
if MEMPROF:
    from memory_profiler import profile
else:
    def profile(f):
        return f

in_filename = None
out_filename = None
if not IN_MEMORY:
    in_filename = "inarray.h5"
    out_filename = "outarray.h5"
    if os.path.exists(in_filename):
        os.remove(in_filename)
    if os.path.exists(out_filename):
        os.remove(out_filename)

rootgrp = None
@profile
def open_datafile(filename):
    global rootgrp
    if IN_MEMORY:
        with open(filename, "rb") as f:
            data = f.read()
        rootgrp = netCDF4.Dataset("memory", mode='r', memory=data)
    else:
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
tslices = np.random.choice(nt - SLICE_THICKNESS, NSLICES)

@profile
def get_slices(data):
    slices = []
    for tslice in tslices:
        slices.append(data[slice(tslice, tslice + SLICE_THICKNESS), :, :])
    return slices
t0 = time()
slices = get_slices(precipitation)
t1 = time()
print("Time for getting %d slices: %.3f" % (NSLICES, (t1 - t0)))

# Time for getting the accumulation
@profile
def get_accum(slices):
    slsum = 0
    for i in range(NSLICES):
        slsum += slices[i].sum()
    return slsum
slsum = get_accum(slices)
t1 = time()
print("Time for summing up %d slices (via netcdf4): %.3f" % (NSLICES, (t1 - t0)))

@profile
def concatenate_slices(slices):
    # HDF5 is handier for outputing datasets
    if IN_MEMORY:
        f = h5py.File(io.BytesIO())
    else:
        f = h5py.File(in_filename, 'w')
    data = f.create_dataset("prec2", shape, chunks=pshape, compression="gzip", compression_opts=1, shuffle=True)
    for i in range(NSLICES):
        data[i * SLICE_THICKNESS: (i + 1) * SLICE_THICKNESS, :, :] = slices[i]
    return data
t0 = time()
prec2 = concatenate_slices(slices)
t1 = time()
print("Time for concatenating %d slices into HDF5 container: %.3f" % (NSLICES, (t1 - t0)))
print(prec2)
#print("cratio", prec2.nbytes / prec2.nbytes_stored)

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

# Compute the accumulation of the random slices into one
@profile
def compute_expr(x):
    scheduler = "single-threaded" if NTHREADS == 1 else "threads"
    with dask.config.set(scheduler=scheduler):
        # I don't see a way on how to spit
        fin = h5py.File(in_filename, 'r')
        x = fin['prec2']
        dx = da.from_array(x)
        res = (np.sin(dx) - 3.2) * (np.cos(dx) + 1.2)
        return da.to_hdf5(out_filename, "/prec2_computed", res)
t0 = time()
b2 = compute_expr(prec2)
t1 = time()
sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"
print("Time for computing '%s' expression (via dask + HDF5): %.3f" % (sexpr, (t1 - t0)))
