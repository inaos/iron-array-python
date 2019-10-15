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
# CLEVEL = 1
# CLIB = ia.ZSTD

t0 = time()
precipitation = ia.from_file("ia-data/rea6/tot_prec/2018.iarray", load_in_mem=IN_MEMORY)
t1 = time()
print("Time to open file: %.3f" % (t1 - t0))
print("dataset:", precipitation)

# Get a random number of slices
nt, nx, ny = precipitation.shape
tslices = np.random.choice(nt, NSLICES)

t0 = time()
sl = []
for tslice in tslices:
    sl.append(precipitation[slice(tslice, tslice + SLICE_THICKNESS),:,:])
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
t0 = time()
sllist = []
slsum = 0
for i in range(NSLICES):
    for (_, block) in sl[i].iter_read_block((1, nx, ny)):
        slsum += block.sum()
    sllist.append(slsum)
t1 = time()
print("Time for summing up %d slices (via iarray iter): %.3f" % (NSLICES, (t1 - t0)))
print(slsum)

# np.testing.assert_allclose(np.array(slsum1), np.array(sllist))

# # Convert ia views into numpy arrays
# t0 = time()
# xnp = [ia.iarray2numpy(sl[i]) for i in range(NSLICES)]
# t1 = time()
# print("Time for converting %d slices into numpy: %.3f" % (NSLICES, (t1 - t0)))
#
# # Apparently the evaluation engine does not handle views well yet
# # x = [ia.numpy2iarray(xnp[i], pshape=(1, nx, ny)) for i in range(NSLICES)]
# x = [ia.numpy2iarray(xnp[i], pshape=(1, nx, ny)) for i in range(NSLICES)]

t0 = time()
dtshape = ia.dtshape(shape=(NSLICES * SLICE_THICKNESS, nx, ny), pshape=(1, nx, ny), dtype=np.float32)
prec2 = ia.empty(dtshape, clevel=CLEVEL, clib=CLIB, nthreads=NTHREADS)
ixnp = iter(sl)
for i, (_, precip_block) in enumerate(prec2.iter_write_block()):
    if i % SLICE_THICKNESS == 0:
        xnp_ = ia.iarray2numpy(next(ixnp))
    precip_block[:, :, :] = xnp_[i % SLICE_THICKNESS]
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

t0 = time()
slsum2 = 0
for i, (_, prec2_block) in enumerate(prec2.iter_read_block((1, nx, ny))):
    slsum2 += prec2_block.sum()
t1 = time()
print("Time for summing up the concatenated ia container: %.3f" % (t1 - t0))
print(slsum2)

# Compute the accumulation of the random slices into one
prec2np = ia.iarray2numpy(prec2)
print("Size of np array:", prec2np.size * 4 / 2**20, "MB")

sexpr = "(sin(x) - 3.2) * (cos(x) + 1.2)"
# sexpr = "(x - 3.2) * (x + 1.2)"

ne.set_num_threads(NTHREADS)
t0 = time()
res = ne.evaluate(sexpr, local_dict={'x': prec2np})
t1 = time()
print("Time for computing '%s' expression (via numexpr): %.3f" % (sexpr, (t1 - t0)))

# t0 = time()
# res = (np.sin(prec2np) - 3.2) * (np.cos(prec2np) + 1.2)
# # res = (prec2np - 3.2) * (prec2np + 1.2)
# t1 = time()
# print("Time for computing '%s' expression (via numpy): %.3f" % (sexpr, (t1 - t0)))

# Compute the accumulation of the random slices into one
t0 = time()
expr = ia.Expr(eval_flags="iterblock", blocksize=0, nthreads=NTHREADS, clevel=CLEVEL)
expr.bind("x", prec2)
expr.compile(sexpr)
b2 = expr.eval((NSLICES * SLICE_THICKNESS, nx, ny), (1, nx, ny), precipitation.dtype)
t1 = time()
print("Time for computing '%s' expression (via ia.Expr()): %.3f" % (sexpr, (t1 - t0)))

# res2 = ia.iarray2numpy(b2)
#
# np.testing.assert_allclose(res, res2, rtol=1e-6)
