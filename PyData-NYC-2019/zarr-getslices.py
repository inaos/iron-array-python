from time import time
import os
import shutil
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
IN_MEMORY = False

in_urlpath = None
out_urlpath = None
if not IN_MEMORY:
    in_urlpath = "zarr_infile.zarr"
    out_urlpath = "zarr_outfile.zarr"
    if os.path.exists(in_urlpath):
        shutil.rmtree(in_urlpath)
    if os.path.exists(out_urlpath):
        shutil.rmtree(out_urlpath)


MEMPROF = True
if MEMPROF:
    from memory_profiler import profile
else:

    def profile(f):
        return f


compressor = Blosc(cname=CNAME, clevel=CLEVEL, shuffle=Blosc.SHUFFLE)


@profile
def open_datafile(urlpath):
    data = zarr.open(urlpath)
    if IN_MEMORY:
        nt, nx, ny = data.shape
        shape = (nt, nx, ny)
        pshape = (1, nx, ny)
        data2 = zarr.empty(shape=shape, dtype="float32", compressor=compressor, chunks=pshape)
        for i in range(nt):
            data2[i, :, :] = data[i]
        data = data2
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
print("Time for summing up %d slices (via zarr): %.3f" % (NSLICES, (t1 - t0)))


@profile
def concatenate_slices(slices):
    if IN_MEMORY:
        data = zarr.empty(shape=shape, dtype="float32", compressor=compressor, chunks=pshape)
    else:
        data = zarr.open(
            in_urlpath, "w", shape=shape, dtype="float32", compressor=compressor, chunks=pshape
        )
    for i in range(NSLICES):
        data[i * SLICE_THICKNESS : (i + 1) * SLICE_THICKNESS, :, :] = slices[i]
    return data


t0 = time()
prec2 = concatenate_slices(slices)
t1 = time()
print("Time for concatenating %d slices into zarr container: %.3f" % (NSLICES, (t1 - t0)))
print(prec2)
print("cratio", prec2.nbytes / prec2.nbytes_stored)


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

# Compute the accumulation of the random slices into one
@profile
def compute_expr(x):
    scheduler = "single-threaded" if NTHREADS == 1 else "threads"
    with dask.config.set(scheduler=scheduler):
        if IN_MEMORY:
            z2 = zarr.empty(shape, dtype="float32", compressor=compressor, chunks=pshape)
        else:
            x = zarr.open(in_urlpath)
            z2 = zarr.open(
                out_urlpath,
                "w",
                shape=shape,
                chunks=pshape,
                dtype="float32",
                compressor=compressor,
            )
        dx = da.from_zarr(x)
        res = (np.sin(dx) - 3.2) * (np.cos(dx) + 1.2)
        return da.to_zarr(res, z2)


t0 = time()
b2 = compute_expr(prec2)
t1 = time()
sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"
print("Time for computing '%s' expression (via dask + zarr): %.3f" % (sexpr, (t1 - t0)))
