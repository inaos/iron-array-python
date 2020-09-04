from time import time
import numpy as np
import numexpr as ne
import iarray as ia


NSLICES = 50
SLICE_THICKNESS = 10
IN_MEMORY = False
NTHREADS = 4
CLEVEL = 5
CLIB = ia.LZ4
BLOCKSIZE = 0
# CLEVEL = 1
# CLIB = ia.ZSTD

sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"
# sexpr = "(x - 3.2) * (x + 1.2)"


MEMPROF = False
if MEMPROF:
    from memory_profiler import profile
else:
    def profile(f):
        return f


@profile
def open_datafile(filename):
    dataset = ia.load(filename, load_in_mem=IN_MEMORY)
    return dataset
t0 = time()
precipitation = open_datafile("ia-data/rea6/tot_prec/2018.iarray")
t1 = time()
print("Time to open file: %.3f" % (t1 - t0))
print("dataset:", precipitation)

# Get a random number of slices
nt, nx, ny = precipitation.shape
shape = (SLICE_THICKNESS, nx, ny)
pshape = (1, nx, ny)
tslices = np.random.choice(nt - SLICE_THICKNESS, NSLICES)

def get_slice(dataset, i):
    tslice = tslices[i]
    return dataset[slice(tslice, tslice + SLICE_THICKNESS), :, :]

@profile
def compute_slices(dataset):
    cslices = []
    # TODO: this shows a leak in the expr.eval() call
    for i in range(NSLICES):
        sl = get_slice(dataset, i)
        expr = ia.Expr(eval_method="iterblock", blocksize=BLOCKSIZE, nthreads=NTHREADS, clevel=CLEVEL)
        expr.bind("x", sl)
        expr.compile(sexpr)
        out = expr.eval(shape, pshape, sl.dtype)
        cslices.append(out)
    return cslices
t0 = time()
slices_computed = compute_slices(precipitation)
t1 = time()
print("Time for computing %d slices: %.3f" % (NSLICES, (t1 - t0)))

@profile
def sum_slices(slices):
    slsum = []
    for i in range(NSLICES):
        slsum.append(ia.iarray2numpy(slices[i]).sum())
    return slsum
t0 = time()
slsum = sum_slices(slices_computed)
t1 = time()
print("Time for summing up the computed slices (iarray): %.3f" % (t1 - t0))

# Convert the slices into numpy
@profile
def tonumpy(dataset):
    npslices = []
    for i in range(NSLICES):
        sl = get_slice(dataset, i)
        npslices.append(ia.iarray2numpy(sl))
    return npslices
t0 = time()
npslices = tonumpy(precipitation)
t1 = time()
print("Time for converting the slices into numpy: %.3f" % (t1 - t0))

@profile
def compute_numexpr(sexpr, npslices):
    ne.set_num_threads(NTHREADS)
    # So as to avoid the result to be cast to a float64, we use an out param
    npslices_computed = []
    for i in range(NSLICES):
        x = npslices[i]
        out = np.empty(x.shape, x.dtype)
        ne.evaluate(sexpr, local_dict={'x': x}, out=out, casting='unsafe')
        npslices_computed.append(out)
    return npslices_computed
t0 = time()
npslices_computed = compute_numexpr(sexpr, npslices)
t1 = time()
print("Time for computing '%s' expression in slices (via numexpr): %.3f" % (sexpr, (t1 - t0)))

@profile
def sum_npslices(npslices):
    slsum = []
    for i in range(NSLICES):
        slsum.append(npslices[i].sum())
    return slsum
t0 = time()
npslsum = sum_npslices(npslices_computed)
t1 = time()
print("Time for summing up the computed slices (pure numpy): %.3f" % (t1 - t0))

np.testing.assert_allclose(np.array(slsum, dtype=np.float32), np.array(npslsum, dtype=np.float32), rtol=1e-5)
