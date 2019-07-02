import pytest
import iarray as ia
import numpy as np


# Matmul
@pytest.mark.parametrize("ashape, apshape, abshape, bshape, bpshape, bbshape, dtype",
                         [
                             ([100, 100], [20, 20], [23, 32], [100, 100], [30, 30], [32, 23], "double"),
                             ([100, 100], None, [100, 100], [100, 100], None, [100, 100], "float"),
                             ([100, 100],  [40, 40], [23, 32], [100], [12], [32], "double"),
                             ([100, 100], None, [100, 100], [100], None, [100], "float"),
                             ([100, 100], None, [100, 100], [100, 100], [20, 20], [100, 23], "double"),
                             ([100, 100], [20, 20], [80, 100], [100, 100], None, [100, 100], "float"),
                             ([100, 100], None, [100, 100], [100], [5], [100], "double"),
                             ([100, 100], [30, 30], [12, 100], [100], None, [100], "float")
                         ])
def test_matmul(ashape, apshape, abshape, bshape, bpshape, bbshape, dtype):
    asize = int(np.prod(ashape))
    a = ia.linspace(asize, -10, 10, ashape, apshape, dtype)
    an = ia.iarray2numpy(a)

    bsize = int(np.prod(bshape))
    b = ia.linspace(bsize, -10, 10, bshape, bpshape, dtype)
    bn = ia.iarray2numpy(b)

    c = ia.matmul(a, b, abshape, bbshape)
    cn = np.matmul(an, bn)

    cn_2 = ia.iarray2numpy(c)

    np.testing.assert_almost_equal(cn, cn_2)



# Matmul slice
@pytest.mark.parametrize("ashape, apshape, astart, astop, abshape, bshape, bpshape, bstart, bstop, bbshape, dtype",
                         [
                             ([100, 100], [20, 20], [20, 40], [70, 90], [23, 32],
                              [100, 100], [30, 30], [10, 20], [60, 70], [32, 23], "double"),
                             ([100, 100], None, [3, 43], [43, 83], [40, 40],
                              [100, 100], None, [12, 13], [52, 53], [40, 40], "float"),
                             ([100, 100],  [40, 40],  [20, 1], [60, 61], [23, 32],
                              [100], [12], [3], [63], [32], "double"),
                             ([100, 100], None, [32, 32], [52, 62], [20, 30],
                              [100], None, [12], [42], [30], "float"),
                             ([100, 100], None, [43, 23], [93, 93], [50, 70],
                              [100, 100], [20, 20], [12, 42], [82, 82], [70, 23], "double"),
                             ([100, 100], [20, 20], [15, 15], [75, 85], [60, 70],
                              [100, 100], None, [22, 22], [92, 32], [70, 10], "float"),
                             ([100, 100], None, [44, 55], [64, 65], [20, 10],
                              [100], [5], [12], [22], [10], "double"),
                             ([100, 100], [30, 30], [12, 20], [32, 30], [10, 10],
                              [100], None, [25], [35], [10], "float")
                         ])
def test_matmul_slice(ashape, apshape, astart, astop, abshape, bshape, bpshape, bstart, bstop, bbshape, dtype):
    asize = int(np.prod(ashape))
    a = ia.linspace(asize, -10, 10, ashape, apshape, dtype)
    an = ia.iarray2numpy(a)
    aslices = tuple(slice(astart[i], astop[i]) for i in range(len(astart)))
    if len(astart) == 1:
        aslices = aslices[0]
    asl = a[aslices]

    bsize = int(np.prod(bshape))
    b = ia.linspace(bsize, -10, 10, bshape, bpshape, dtype)
    bn = ia.iarray2numpy(b)
    bslices = tuple(slice(bstart[i], bstop[i]) for i in range(len(bstart)))
    if len(bstart) == 1:
        bslices = bslices[0]
    bsl = b[bslices]

    c = ia.matmul(asl, bsl, abshape, bbshape)
    cn = np.matmul(an[aslices], bn[bslices])

    cn_2 = ia.iarray2numpy(c)

    np.testing.assert_almost_equal(cn, cn_2)
