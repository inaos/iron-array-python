from time import time
import os
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

MEMPROF = True
if MEMPROF:
    from memory_profiler import profile
else:
    def profile(f):
        return f

in_filename = None
out_filename = None
if not IN_MEMORY:
    in_filename = "inarray.iarray"
    out_filename = "outarray.iarray"
    if os.path.exists(in_filename):
        os.remove(in_filename)
    if os.path.exists(out_filename):
        os.remove(out_filename)

@profile
def open_datafile(filename):
    dataset = ia.from_file(filename, load_in_mem=IN_MEMORY)
    return dataset
t0 = time()
precipitation = open_datafile("ia-data/rea6/tot_prec/2018.iarray")
t1 = time()
print("Time to open file: %.3f" % (t1 - t0))
print("dataset:", precipitation)

# Get a random number of slices
nt, nx, ny = precipitation.shape
shape = (NSLICES * SLICE_THICKNESS, nx, ny)
pshape = (1, nx, ny)
tslices = np.random.choice(nt - SLICE_THICKNESS, NSLICES)

@profile
def get_slices(dataset):
    sl = []
    for tslice in tslices:
        sl.append(dataset[slice(tslice, tslice + SLICE_THICKNESS),:,:])
    return sl
t0 = time()
slices = get_slices(precipitation)
t1 = time()
print("Time for getting %d slices: %.3f" % (NSLICES, (t1 - t0)))


# # Time for getting the accumulation
# t0 = time()
# slsum1 = []
# for i in range(NSLICES):
#     slsum1.append(ia.iarray2numpy(sl[i]).sum())
# t1 = time()
# print("Time for summing out %d slices (via numpy): %.3f" % (NSLICES, (t1 - t0)))

# Time for getting the accumulation
@profile
def get_accum(sl):
    slsum = 0
    for i in range(NSLICES):
        for (_, block) in sl[i].iter_read_block((1, nx, ny)):
            slsum += block.sum()
    return slsum
t0 = time()
slsum = get_accum(slices)
t1 = time()
print("Time for summing up %d slices (via iarray iter): %.3f" % (NSLICES, (t1 - t0)))

@profile
def concatenate_slices(slices):
    dtshape = ia.dtshape(shape=shape, pshape=pshape, dtype=np.float32)
    iarr = ia.empty(dtshape, clevel=CLEVEL, clib=CLIB, nthreads=NTHREADS, blocksize=BLOCKSIZE, filename=in_filename)
    islices = iter(slices)
    for i, (_, precip_block) in enumerate(iarr.iter_write_block()):
        if i % SLICE_THICKNESS == 0:
            slice_np = ia.iarray2numpy(next(islices))
        precip_block[:, :, :] = slice_np[i % SLICE_THICKNESS]
    return iarr
t0 = time()
prec2 = concatenate_slices(slices)
t1 = time()
print("Time for concatenating %d slices into ia container: %.3f" % (NSLICES, (t1 - t0)))
print(prec2)
# prec2 = ia.linspace(dtshape, 0, 10, clevel=CLEVEL, nthreads=NTHREADS)
print("cratio", prec2.cratio)

# # Compute the accumulation of the random slices into one
# t0 = time()
# accum = sl[0]
# for i in range(1, NSLICES):
#     accum += sl[i]
# result = accum.eval()
# t1 = time()
# print("Time for accumulating %d slices into one (via eval()): %.3f" % (NSLICES, (t1 - t0)))

@profile
def sum_concat(iarr):
    concatsum = 0
    for (_, iarr_block) in iarr.iter_read_block():
        concatsum += iarr_block.sum()
    return concatsum
t0 = time()
concatsum = sum_concat(prec2)
t1 = time()
print("Time for summing up the concatenated ia container: %.3f" % (t1 - t0))

# Compute the accumulation of the random slices into one
@profile
def tonumpy(iarr):
    return ia.iarray2numpy(iarr)
t0 = time()
prec2np = tonumpy(prec2)
t1 = time()
print("Time for converting the concatenated ia container into numpy: %.3f" % (t1 - t0))
print("Size of numpy array:", prec2np.size * 4 / 2**20, "MB")

sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"
# sexpr = "(x - 3.2) * (x + 1.2)"

@profile
def compute_numexpr(sexpr, x):
    ne.set_num_threads(NTHREADS)
    # So as to avoid the result to be cast to a float64, we use an out param
    out = np.empty(x.shape, x.dtype)
    return ne.evaluate(sexpr, local_dict={'x': x}, out=out, casting='unsafe')
t0 = time()
res = compute_numexpr(sexpr, prec2np)
t1 = time()
print("Time for computing '%s' expression (via numexpr): %.3f" % (sexpr, (t1 - t0)))

# t0 = time()
# res = (np.sin(prec2np) - 3.2) * (np.cos(prec2np) + 1.2)
# # res = (prec2np - 3.2) * (prec2np + 1.2)
# t1 = time()
# print("Time for computing '%s' expression (via numpy): %.3f" % (sexpr, (t1 - t0)))

# Compute the accumulation of the random slices into one
@profile
def compute_expr(sexpr, x):
    expr = ia.Expr(eval_flags="iterblock", blocksize=BLOCKSIZE, nthreads=NTHREADS, clevel=CLEVEL)
    if not IN_MEMORY:
        x = ia.from_file(in_filename)
    expr.bind("x", x)
    expr.compile(sexpr)
    return expr.eval(shape, pshape, precipitation.dtype, filename=out_filename)
t0 = time()
b2 = compute_expr(sexpr, prec2)
t1 = time()
print("Time for computing '%s' expression (via ia.Expr()): %.3f" % (sexpr, (t1 - t0)))

# res2 = ia.iarray2numpy(b2)
# np.testing.assert_allclose(res, res2, rtol=1e-5)
