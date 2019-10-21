from time import time
import numpy as np
import zarr
import dask.array as da
from numcodecs import Blosc
import dask


NTHREADS = 4
CLEVEL = 5
CNAME = "lz4"
# CLEVEL = 1
# CNAME = zarr.ZSTD

NSLICES = 50
SLICE_THICKNESS = 10

MEMPROF = True
if MEMPROF:
    from memory_profiler import profile
else:
    def profile(f):
        return f

compressor = Blosc(cname=CNAME, clevel=CLEVEL, shuffle=Blosc.SHUFFLE)

@profile
def open_datafile(filename):
    data = zarr.open(filename)
    return data
t0 = time()
precipitation = open_datafile("ia-data/rea6/tot_prec/2018.zarr")
t1 = time()
print("Time to open file: %.3f" % (t1 - t0))
# print("dataset:", precipitation)

# Get a random number of slices
nt, nx, ny = precipitation.shape
shape = (NSLICES * SLICE_THICKNESS, nx, ny)
pshape = (1, nx, ny)
# tslices = np.random.choice(nt - SLICE_THICKNESS, NSLICES)
np.random.seed(1)
# np.random.seed(3)
tslice = np.random.choice(nt - NSLICES * SLICE_THICKNESS)

@profile
def concatenate_slices(dataset):
    data = zarr.empty(shape=shape, dtype=dataset.dtype, compressor=compressor, chunks=pshape)
    for i in range(NSLICES * SLICE_THICKNESS):
        data[i] = dataset[i]
    return data
t0 = time()
prec2 = concatenate_slices(precipitation)
t1 = time()
print("Time for concatenating %d slices into a zarr container: %.3f" % (NSLICES, (t1 - t0)))
print(prec2)
print("cratio", prec2.nbytes / prec2.nbytes_stored)

# Compute the accumulation of the random slices into one
@profile
def compute_expr(x):
    scheduler = "single-threaded" if NTHREADS == 1 else "threads"
    with dask.config.set(scheduler=scheduler):
        z2 = zarr.empty(shape, dtype=x.dtype, compressor=compressor, chunks=pshape)
        dx = da.from_zarr(x)
        res = (np.sin(dx) - 3.2) * (np.cos(dx) + 1.2)
        return da.to_zarr(res, z2)
t0 = time()
b2 = compute_expr(prec2)
t1 = time()
sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"
print("Time for computing '%s' expression (via dask + zarr): %.3f" % (sexpr, (t1 - t0)))

@profile
def sum_concat(data):
    concatsum = 0
    for i in range(len(data)):
        concatsum += data[i].sum()
    return concatsum
t0 = time()
concatsum = sum_concat(prec2)
t1 = time()
print("Time for summing up the concatenated zarr container: %.3f" % (t1 - t0))
