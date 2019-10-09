from time import time
import numpy as np
import iarray as ia


NSLICES = 100
SLICE_THICKNESS = 10
IN_MEMORY = False

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
slsum2 = []
for i in range(NSLICES):
    slsum = 0
    for (_, block) in sl[i].iter_read_block((SLICE_THICKNESS, nx, ny)):
        slsum += block.sum()
    slsum2.append(slsum)
t1 = time()
print("Time for summing up %d slices (via iarray iter): %.3f" % (NSLICES, (t1 - t0)))

# np.testing.assert_allclose(np.array(slsum1), np.array(slsum2))
