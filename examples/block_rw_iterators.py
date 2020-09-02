# Using block iterators.  iter_read_block() reads blocks whereas iter_write_block() assign blocks to destination
# Here we are using plainbuffers for simplicity

import iarray as ia
from itertools import zip_longest as izip


# Create an empty container for filling it with another one
dtshape = ia.dtshape(shape=[10, 10])
c1 = ia.empty(dtshape)
c2 = ia.arange(dtshape)

bshape = [4, 5]
for i, ((_, p1), (_, p2)) in enumerate(izip(c2.iter_read_block(bshape), c1.iter_write_block(bshape))):
    p2[:] = p1

# Convert back into a numpy array and print the results
c3 = ia.iarray2numpy(c1)
print(c3)
