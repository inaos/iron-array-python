import pytest
import iarray as ia
import numpy as np


# Matmul
@pytest.mark.parametrize(
    "ashape, achunkshape, ablockshape,"
    "bshape, bchunkshape, bblockshape,"
    "cchunkshape, cblockshape, dtype",
    [
        ([20, 20], [10, 10], [5, 5], [20, 20], [10, 10], [5, 5], [10, 10], [5, 5], np.float64),
        ([100, 100], None, None, [100, 100], None, None, None, None, np.float32),
        ([100, 100], [40, 40], [12, 12], [100], [60], [30], [50], [30], np.float64),
        ([100, 100], None, None, [100], None, None, None, None, np.float32),
        ([100, 100], None, None, [100, 100], [20, 20], [12, 3], [70, 23], [25, 5], np.float64),
        ([100, 100], [20, 20], [7, 9], [100, 100], None, None, None, None, np.float32),
        ([100, 100], None, None, [100], [50], [25], [100], [49], np.float64),
        ([500, 100], [30, 30], [5, 20], [100], None, None, [200], [100], np.float32),
    ],
)
def test_matmul(
    ashape,
    achunkshape,
    ablockshape,
    bshape,
    bchunkshape,
    bblockshape,
    cchunkshape,
    cblockshape,
    dtype,
):
    if achunkshape is None:
        astorage = ia.StorageProperties(plainbuffer=True)
    else:
        astorage = ia.StorageProperties(achunkshape, ablockshape)
    a = ia.linspace(ia.dtshape(ashape, dtype), -10, 1, storage=astorage)
    an = ia.iarray2numpy(a)

    if bchunkshape is None:
        bstorage = ia.StorageProperties(plainbuffer=True)
    else:
        bstorage = ia.StorageProperties(bchunkshape, bblockshape)
    b = ia.linspace(ia.dtshape(bshape, dtype), -1, 10, storage=bstorage)
    bn = ia.iarray2numpy(b)

    if cchunkshape is None:
        cstorage = ia.StorageProperties(plainbuffer=True)
    else:
        cstorage = ia.StorageProperties(cchunkshape, cblockshape)

    c = ia.matmul(a, b, storage=cstorage)
    cn_2 = ia.iarray2numpy(c)

    cn = np.matmul(an, bn)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(cn, cn_2, rtol=rtol)


# Matmul slice
@pytest.mark.parametrize(
    "ashape, achunkshape, ablockshape, astart, astop,"
    "bshape, bchunkshape, bblockshape, bstart, bstop,"
    "cchunkshape, cblockshape, dtype",
    [
        (
            [100, 100],
            [20, 20],
            [10, 12],
            [20, 40],
            [70, 90],
            [100, 100],
            [30, 30],
            [15, 4],
            [10, 20],
            [60, 70],
            [20, 25],
            [10, 10],
            np.float64,
        ),
        (
            [100, 100],
            None,
            None,
            [3, 43],
            [43, 83],
            [100, 200],
            None,
            None,
            [12, 13],
            [52, 153],
            None,
            None,
            np.float32,
        ),
        (
            [100, 100],
            [40, 40],
            [12, 7],
            [20, 1],
            [60, 61],
            [100],
            [44],
            [22],
            [3],
            [63],
            None,
            None,
            np.float64,
        ),
        (
            [100, 100],
            None,
            None,
            [12, 32],
            [82, 62],
            [100],
            None,
            None,
            [12],
            [42],
            [30],
            [10],
            np.float32,
        ),
        (
            [100, 100],
            None,
            None,
            [43, 23],
            [93, 93],
            [100, 100],
            [20, 20],
            [20, 2],
            [12, 42],
            [82, 82],
            [20, 20],
            [8, 11],
            np.float64,
        ),
        (
            [100, 100],
            [20, 20],
            [5, 6],
            [15, 15],
            [75, 85],
            [100, 100],
            None,
            None,
            [22, 22],
            [92, 32],
            None,
            None,
            np.float32,
        ),
        (
            [1000, 500],
            None,
            None,
            [144, 55],
            [964, 465],
            [500],
            [200],
            [90],
            [12],
            [422],
            [160],
            [50],
            np.float64,
        ),
        (
            [1000, 1000],
            [300, 300],
            [12, 50],
            [12, 20],
            [320, 300],
            [2000],
            None,
            None,
            [140],
            [420],
            [100],
            [40],
            np.float32,
        ),
    ],
)
def test_matmul_slice(
    ashape,
    achunkshape,
    ablockshape,
    astart,
    astop,
    bshape,
    bchunkshape,
    bblockshape,
    bstart,
    bstop,
    cchunkshape,
    cblockshape,
    dtype,
):
    if achunkshape is None:
        astorage = ia.StorageProperties(plainbuffer=True)
    else:
        astorage = ia.StorageProperties(achunkshape, ablockshape)
    a = ia.linspace(ia.dtshape(ashape, dtype), -1, -2, storage=astorage)
    an = ia.iarray2numpy(a)
    aslices = tuple(slice(astart[i], astop[i]) for i in range(len(astart)))
    if len(astart) == 1:
        aslices = aslices[0]
    asl = a[aslices]

    if bchunkshape is None:
        bstorage = ia.StorageProperties(plainbuffer=True)
    else:
        bstorage = ia.StorageProperties(bchunkshape, bblockshape)
    b = ia.linspace(ia.dtshape(bshape, dtype), 1, 200, storage=bstorage)
    bn = ia.iarray2numpy(b)
    bslices = tuple(slice(bstart[i], bstop[i]) for i in range(len(bstart)))
    if len(bstart) == 1:
        bslices = bslices[0]
    bsl = b[bslices]

    if cchunkshape is None:
        cstorage = ia.StorageProperties(plainbuffer=True)
    else:
        cstorage = ia.StorageProperties(cchunkshape, cblockshape)

    c = ia.matmul(asl, bsl, storage=cstorage)
    cn = np.matmul(an[aslices], bn[bslices])

    cn_2 = ia.iarray2numpy(c)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(cn, cn_2, rtol=rtol)
