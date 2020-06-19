import pytest
import iarray as ia
import numpy as np


# Matmul
@pytest.mark.parametrize("ashape, apshape, abshape, bshape, bpshape, bbshape, dtype",
                         [
                             ([20, 20], [10, 10], [10, 10], [20, 20], [10, 10], [10, 10], np.float64),
                             ([100, 100], None, [100, 100], [100, 100], None, [100, 100], np.float32),
                             ([100, 100], [40, 40], [23, 32], [100], [12], [32], np.float64),
                             ([100, 100], None, [100, 100], [100], None, [100], np.float32),
                             ([100, 100], None, [100, 100], [100, 100], [20, 20], [100, 23], np.float64),
                             ([100, 100], [20, 20], [80, 100], [100, 100], None, [100, 100], np.float32),
                             ([100, 100], None, [100, 100], [100], [5], [100], np.float64),
                             ([100, 100], [30, 30], [12, 100], [100], None, [100], np.float32)
                         ])
def test_matmul(ashape, apshape, abshape, bshape, bpshape, bbshape, dtype):
    if apshape is None:
        astorage = ia.StorageProperties("plainbuffer")
    else:
        astorage = ia.StorageProperties("blosc", apshape, apshape, False)
    a = ia.linspace(ia.dtshape(ashape, dtype), -10, 1, storage=astorage)
    an = ia.iarray2numpy(a)

    if bpshape is None:
        bstorage = ia.StorageProperties("plainbuffer")
    else:
        bstorage = ia.StorageProperties("blosc", bpshape, bpshape, False)
    b = ia.linspace(ia.dtshape(bshape, dtype), -1, 10, storage=bstorage)
    bn = ia.iarray2numpy(b)

    if abshape is None and bbshape is None:
        cstorage = ia.StorageProperties("plainbuffer")
    else:
        if len(bbshape) == 2:
            cpshape = [a.shape[0], b.shape[1]]
            if abshape is not None:
                cpshape[0] = abshape[0]
            if bbshape is not None:
                cpshape[1] = bbshape[1]
            cpshape = tuple(cpshape)
        else:
            cpshape = [a.shape[0]]
            if abshape is not None:
                cpshape[0] = abshape[0]
            cpshape = tuple(cpshape)
        cstorage = ia.StorageProperties("blosc", cpshape, cpshape)

    c = ia.matmul(a, b, abshape, bbshape, storage=cstorage)
    cn_2 = ia.iarray2numpy(c)
    print(cn_2)
    cn = np.matmul(an, bn)


    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(cn, cn_2, rtol=rtol)


# Matmul slice
@pytest.mark.parametrize(
    "ashape, apshape, astart, astop, abshape, bshape, bpshape, bstart, bstop, bbshape, dtype",
    [
        ([100, 100], [20, 20], [20, 40], [70, 90], [23, 32],
         [100, 100], [30, 30], [10, 20], [60, 70], [32, 23], np.float64),
        ([100, 100], None, [3, 43], [43, 83], [40, 40],
         [100, 100], None, [12, 13], [52, 53], [40, 40], np.float32),
        ([100, 100], [40, 40], [20, 1], [60, 61], [23, 32],
         [100], [12], [3], [63], [32], np.float64),
        ([100, 100], None, [32, 32], [52, 62], [20, 30],
         [100], None, [12], [42], [30], np.float32),
        ([100, 100], None, [43, 23], [93, 93], [50, 70],
         [100, 100], [20, 20], [12, 42], [82, 82], [70, 23], np.float64),
        ([100, 100], [20, 20], [15, 15], [75, 85], [60, 70],
         [100, 100], None, [22, 22], [92, 32], [70, 10], np.float32),
        ([100, 100], None, [44, 55], [64, 65], [20, 10],
         [100], [5], [12], [22], [10], np.float64),
        ([100, 100], [30, 30], [12, 20], [32, 30], [10, 10],
         [100], None, [25], [35], [10], np.float32)
    ])
def test_matmul_slice(ashape, apshape, astart, astop, abshape, bshape, bpshape,
                      bstart, bstop, bbshape, dtype):
    if apshape is None:
        astorage = ia.StorageProperties("plainbuffer")
    else:
        astorage = ia.StorageProperties("blosc", apshape, apshape, False)
    a = ia.linspace(ia.dtshape(ashape, dtype), -1, -2, storage=astorage)
    an = ia.iarray2numpy(a)
    aslices = tuple(slice(astart[i], astop[i]) for i in range(len(astart)))
    if len(astart) == 1:
        aslices = aslices[0]
    asl = a[aslices]

    if bpshape is None:
        bstorage = ia.StorageProperties("plainbuffer")
    else:
        bstorage = ia.StorageProperties("blosc", bpshape, bpshape, False)
    b = ia.linspace(ia.dtshape(bshape, dtype), 1, 200, storage=bstorage)
    bn = ia.iarray2numpy(b)
    bslices = tuple(slice(bstart[i], bstop[i]) for i in range(len(bstart)))
    if len(bstart) == 1:
        bslices = bslices[0]
    bsl = b[bslices]

    if abshape is None and bbshape is None:
        cstorage = ia.StorageProperties("plainbuffer")
    else:
        if len(bbshape) == 2:
            cpshape = [a.shape[0], b.shape[1]]
            if abshape is not None:
                cpshape[0] = abshape[0]
            if bbshape is not None:
                cpshape[1] = bbshape[1]
            cpshape = tuple(cpshape)
        else:
            cpshape = [a.shape[0]]
            if abshape is not None:
                cpshape[0] = abshape[0]
            cpshape = tuple(cpshape)
        cstorage = ia.StorageProperties("blosc", cpshape, cpshape)

    c = ia.matmul(asl, bsl, abshape, bbshape, storage=cstorage)
    cn = np.matmul(an[aslices], bn[bslices])

    cn_2 = ia.iarray2numpy(c)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(cn, cn_2, rtol=rtol)
