# Examples on getting slices

import iarray as ia
import numpy as np


ashape, astart, astop = [100, 100], [20, 40], [70, 90]
bshape, bstart, bstop = [100, 100], [10, 20], [60, 70]
dtype = np.float64

asize = int(np.prod(ashape))
a = ia.linspace(ia.DTShape(ashape, dtype), -10, 10)
an = ia.iarray2numpy(a)
aslices = tuple(slice(astart[i], astop[i]) for i in range(len(astart)))
if len(astart) == 1:
    aslices = aslices[0]
asl = a[aslices]
print(ia.iarray2numpy(asl))

bsize = int(np.prod(bshape))
b = ia.linspace(ia.DTShape(bshape, dtype), -10, 10)
bn = ia.iarray2numpy(b)
bslices = tuple(slice(bstart[i], bstop[i]) for i in range(len(bstart)))
if len(bstart) == 1:
    bslices = bslices[0]
bsl = b[bslices]
print(ia.iarray2numpy(bsl))
