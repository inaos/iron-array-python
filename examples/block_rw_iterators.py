# Using block iterators.  iter_read_block() reads blocks whereas iter_write_block() assign blocks to destination.
# We use a plainbuffer array in destination, as it is the only one that supports assignments.
# Please note that, as the `iter_read_block` finish first, we need to use zip_longest so as to complete the copy.

import numpy as np
import iarray as ia
from itertools import zip_longest


# Create an empty container for filling it with another one
dtshape = ia.dtshape(shape=[10, 10])
c1 = ia.empty(dtshape, storage=ia.StorageProperties(plainbuffer=True))
c2 = ia.arange(dtshape)

bshape = [4, 5]
for i, ((_, p1), (_, p2)) in enumerate(zip_longest(c2.iter_read_block(bshape), c1.iter_write_block(bshape))):
    p2[:] = p1

# Convert back into a numpy array and print the results
c3 = ia.iarray2numpy(c1)
print(c3)

c0 = np.arange(c3.size).reshape(c3.shape)
np.testing.assert_allclose(c0, c3)
