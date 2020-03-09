from time import time
import numpy as np
import numexpr as ne
import iarray as ia
from numba import jit


NSLICES = 50
SLICE_THICKNESS = 10
NTHREADS = 4
CLEVEL = 5
CLIB = ia.LZ4
BLOCKSIZE = 0
# CLEVEL = 1
# CLIB = ia.ZSTD

sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"
# sexpr = "(x - 3.2) * (x + 1.2)"


MEMPROF = True
if MEMPROF:
    from memory_profiler import profile
else:
    def profile(f):
        return f


@profile
def open_datafile(filename):
    dataset = ia.load(filename, load_in_mem=False)
    return dataset
t0 = time()
precipitation = open_datafile("ia-data/rea6/tot_prec/2018.iarray")
t1 = time()
print("Time to open file: %.3f" % (t1 - t0))
#print("dataset:", precipitation)

# Get a random number of slices
nt, nx, ny = precipitation.shape
shape = (NSLICES * SLICE_THICKNESS, nx, ny)
pshape = (1, nx, ny)
# tslices = np.random.choice(nt - SLICE_THICKNESS, NSLICES)
np.random.seed(1)
# np.random.seed(3)
tslice = np.random.choice(nt - NSLICES * SLICE_THICKNESS)

def get_slice(dataset, i):
    return dataset[slice(tslice + i, tslice + i + 1), :, :]

@profile
def concatenate_slices(dataset):
    dtshape = ia.dtshape(shape=shape, pshape=pshape, dtype=np.float32)
    iarr = ia.empty(dtshape, clevel=CLEVEL, clib=CLIB, nthreads=NTHREADS, blocksize=BLOCKSIZE)
    for i, (_, precip_block) in enumerate(iarr.iter_write_block()):
        slice_np = ia.iarray2numpy(get_slice(dataset, i))
        precip_block[:, :, :] = slice_np
    return iarr
t0 = time()
prec2 = concatenate_slices(precipitation)
t1 = time()
print("Time for concatenating %d slices into an ia container: %.3f" % (NSLICES, (t1 - t0)))
#print(prec2)
# prec2 = ia.linspace(dtshape, 0, 10, clevel=CLEVEL, nthreads=NTHREADS)
print("cratio", prec2.cratio)

@profile
def compute_slices(dataset):
    expr = ia.Expr(eval_flags="iterblock", clevel=CLEVEL, clib=CLIB, nthreads=NTHREADS, blocksize=BLOCKSIZE)
    expr.bind("x", dataset)
    expr.compile(sexpr)
    out = expr.eval(shape, pshape, dataset.dtype)
    return out
t0 = time()
slices_computed = compute_slices(prec2)
t1 = time()
print("Time for computing %d slices (iarray): %.3f" % (NSLICES, (t1 - t0)))

@profile
def sum_slices(slice):
    slsum = ia.iarray2numpy(slice).sum()
    return slsum
t0 = time()
slsum = sum_slices(slices_computed)
t1 = time()
print("Time for summing up the computed slices (iarray): %.3f" % (t1 - t0))

# Convert the slices into numpy
t0 = time()
npslices = ia.iarray2numpy(prec2)
t1 = time()
print("Time for converting the slices into numpy: %.3f" % (t1 - t0))

@profile
def compute_numpy(x):
    out = (np.sin(x) - 3.2) * (np.cos(x) + 1.2)
    return out
t0 = time()
npslices_computed = compute_numpy(npslices)
t1 = time()
print("Time for computing '%s' expression in slices (via numpy): %.3f" % (sexpr, (t1 - t0)))

@profile
def sum_npslices(npslices):
    return npslices.sum()
t0 = time()
npslsum = sum_npslices(npslices_computed)
t1 = time()
print("Time for summing up the computed slices (pure numpy): %.3f" % (t1 - t0))

np.testing.assert_allclose(np.array(slsum, dtype=np.float32), np.array(npslsum, dtype=np.float32), rtol=1e-5)

@profile
def compute_numexpr(sexpr, x):
    ne.set_num_threads(NTHREADS)
    # So as to avoid the result to be cast to a float64, we use an out param
    out = np.empty(x.shape, x.dtype)
    ne.evaluate(sexpr, local_dict={'x': x}, out=out, casting='unsafe')
    return out
t0 = time()
npslices_computed2 = compute_numexpr(sexpr, npslices)
t1 = time()
print("Time for computing '%s' expression in slices (via numexpr): %.3f" % (sexpr, (t1 - t0)))

@jit(nopython=True, cache=True)
def poly_numba2(x, y):
    nt, nx, ny = x.shape
    for i in range(nt):
        for j in range(nx):
            for k in range(ny):
                y[i,j,k] = (np.sin(x[i,j,k]) - 3.2) * (np.cos(x[i,j,k]) + 1.2)

@profile
def compute_numba(x):
    # So as to avoid the result to be cast to a float64
    out = np.empty(x.shape, x.dtype)
    poly_numba2(x, out)
    return out
t0 = time()
npslices_computed3 = compute_numba(npslices)
t1 = time()
print("Time for computing '%s' expression in slices (via numba): %.3f" % (sexpr, (t1 - t0)))
