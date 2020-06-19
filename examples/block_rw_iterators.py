import iarray as ia
from itertools import zip_longest as izip


# Create an empty container for filling it with another one
dtshape = ia.dtshape(shape=[10, 10], pshape=[4, 5])
c1 = ia.empty(dtshape)
c2 = ia.arange(dtshape)

for i, ((_, p1), (_, p2)) in enumerate(izip(c2.iter_read_block(dtshape.pshape), c1.iter_write_block())):
    p2[:] = p1

c3 = ia.iarray2numpy(c1)

print(c3)
