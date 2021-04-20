# Using block iterators.  iter_read_block() reads blocks whereas iter_write_block() assign blocks to destination.
# We use a plainbuffer array in destination, as it is the only one that supports assignments.
# Please note that, as the `iter_read_block` finishes first, we need to use zip_longest so as to complete the copy.

import numpy as np
import iarray as ia
from itertools import zip_longest


# Create an empty container for filling it with another one
# Note how dtypes can be different.
shape = (10, 100)

c1 = ia.arange(shape, dtype=np.float64)
c2 = ia.empty(shape, dtype=np.float32, plainbuffer=True)

for i, ((_, p1), (_, p2)) in enumerate(zip_longest(c1.iter_read_block(), c2.iter_write_block())):
    p2[:] = p1

# Convert back into a numpy array and print the results
c3 = ia.iarray2numpy(c2)
print(c3[:10, :10], c2.dtype)

c0 = np.arange(c3.size).reshape(c3.shape)
np.testing.assert_allclose(c0, c3)
