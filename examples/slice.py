import iarray as ia
import numpy as np

cfg = ia.Config()
ctx = ia.Context(cfg)

ashape, apshape, astart, astop, abshape, bshape, bpshape, bstart, bstop, bbshape, dtype = ([100, 100], [20, 20], [20, 40], [70, 90], [23, 32],
                              [100, 100], [30, 30], [10, 20], [60, 70], [32, 23], "double")

asize = int(np.prod(ashape))
a = ia.linspace(ctx, asize, -10, 10, ashape, apshape, dtype)
an = ia.iarray2numpy(ctx, a)
aslices = tuple(slice(astart[i], astop[i]) for i in range(len(astart)))
if len(astart) == 1:
    aslices = aslices[0]
asl = a[aslices]

bsize = int(np.prod(bshape))
b = ia.linspace(ctx, bsize, -10, 10, bshape, bpshape, dtype)
bn = ia.iarray2numpy(ctx, b)
bslices = tuple(slice(bstart[i], bstop[i]) for i in range(len(bstart)))
if len(bstart) == 1:
    bslices = bslices[0]
bsl = b[bslices]

c = ia.matmul(ctx, asl, bsl, abshape, bbshape)
cn = np.matmul(an[aslices], bn[bslices])

cn_2 = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn_2)