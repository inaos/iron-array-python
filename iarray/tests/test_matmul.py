import pytest
import iarray as ia
import numpy as np


# Matmul
@pytest.mark.parametrize(
    "ashape, achunks, ablocks, acontiguous, aurlpath,"
    "bshape, bchunks, bblocks, bcontiguous, burlpath,"
    "cchunks, cblocks, dtype, ccontiguous, curlpath",
    [
        (
            [20, 20],
            [10, 10],
            [5, 5],
            True,
            None,
            [20, 20],
            [10, 10],
            [5, 5],
            True,
            None,
            [10, 10],
            [5, 5],
            np.float64,
            True,
            None,
        ),
        (
            [100, 100],
            [100, 100],
            [10, 10],
            False,
            "test_matmul_asparse.iarr",
            [100, 100],
            [10, 10],
            [10, 10],
            False,
            "test_matmul_bsparse.iarr",
            [20, 20],
            [10, 10],
            np.float32,
            False,
            "test_matmul_csparse.iarr",
        ),
        (
            [100, 100],
            [40, 40],
            [12, 12],
            False,
            None,
            [100],
            [60],
            [30],
            False,
            None,
            [50],
            [30],
            np.float64,
            False,
            None,
        ),
        (
            [100, 100],
            [50, 50],
            [20, 20],
            True,
            "test_matmul_acontiguous.iarr",
            [100],
            [50],
            [20],
            True,
            "test_matmul_bcontiguous.iarr",
            [30],
            [15],
            np.float32,
            True,
            "test_matmul_ccontiguous.iarr",
        ),
        (
            [100, 100],
            [77, 12],
            [23, 12],
            True,
            "test_matmul_acontiguous.iarr",
            [100, 100],
            [20, 20],
            [12, 3],
            False,
            None,
            [70, 23],
            [25, 5],
            np.float64,
            False,
            "test_matmul_csparse.iarr",
        ),
        (
            [100, 100],
            [20, 20],
            [7, 9],
            False,
            None,
            [100, 100],
            [50, 45],
            [20, 20],
            True,
            "test_matmul_bcontiguous.iarr",
            [40, 27],
            [20, 20],
            np.float32,
            True,
            "test_matmul_ccontiguous.iarr",
        ),
        (
            [100, 100],
            [10, 10],
            [10, 10],
            True,
            "test_matmul_acontiguous",
            [100],
            [50],
            [25],
            False,
            None,
            [100],
            [49],
            np.float64,
            False,
            None,
        ),
        (
            [500, 100],
            [30, 30],
            [5, 20],
            False,
            "test_matmul_asparse.iarr",
            [100],
            [50],
            [25],
            False,
            None,
            [200],
            [100],
            np.float32,
            True,
            "test_matmul_ccontiguous.iarr",
        ),
    ],
)
def test_matmul(
    ashape,
    achunks,
    ablocks,
    bshape,
    bchunks,
    bblocks,
    cchunks,
    cblocks,
    dtype,
    acontiguous,
    aurlpath,
    bcontiguous,
    burlpath,
    ccontiguous,
    curlpath,
):

    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)
    ia.remove_urlpath(curlpath)
    acfg = ia.Config(chunks=achunks, blocks=ablocks, contiguous=acontiguous, urlpath=aurlpath)
    a = ia.linspace(ashape, -10, 1, dtype=dtype, cfg=acfg)
    an = ia.iarray2numpy(a)

    bcfg = ia.Config(chunks=bchunks, blocks=bblocks, contiguous=bcontiguous, urlpath=burlpath)
    b = ia.linspace(bshape, -1, 10, dtype=dtype, cfg=bcfg)
    bn = ia.iarray2numpy(b)

    ccfg = ia.Config(chunks=cchunks, blocks=cblocks, contiguous=ccontiguous, urlpath=curlpath)
    c = ia.matmul(a, b, cfg=ccfg)
    cn_2 = ia.iarray2numpy(c)

    cn = np.matmul(an, bn)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(cn, cn_2, rtol=rtol)

    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)
    ia.remove_urlpath(curlpath)


# Matmul slice
@pytest.mark.parametrize(
    "ashape, achunks, ablocks, astart, astop, acontiguous, aurlpath,"
    "bshape, bchunks, bblocks, bstart, bstop,bcontiguous, burlpath,"
    "cchunks, cblocks, dtype, ccontiguous, curlpath,",
    [
        (
            [100, 100],
            [20, 20],
            [10, 12],
            [20, 40],
            [70, 90],
            True,
            None,
            [100, 100],
            [30, 30],
            [15, 4],
            [10, 20],
            [60, 70],
            True,
            None,
            [20, 25],
            [10, 10],
            np.float64,
            True,
            None,
        ),
        (
            [100, 100],
            None,
            None,
            [3, 43],
            [43, 83],
            False,
            None,
            [100, 200],
            None,
            None,
            [12, 13],
            [52, 153],
            False,
            None,
            None,
            None,
            np.float32,
            False,
            None,
        ),
        (
            [100, 100],
            [40, 40],
            [12, 7],
            [20, 1],
            [60, 61],
            True,
            "test_matmul_acontiguous.iarr",
            [100],
            [44],
            [22],
            [3],
            [63],
            True,
            "test_matmul_bcontiguous.iarr",
            None,
            None,
            np.float64,
            True,
            "test_matmul_ccontiguous.iarr",
        ),
        (
            [100, 100],
            None,
            None,
            [12, 32],
            [82, 62],
            False,
            "test_matmul_asparse.iarr",
            [100],
            None,
            None,
            [12],
            [42],
            False,
            "test_matmul_bsparse.iarr",
            [30],
            [10],
            np.float32,
            False,
            "test_matmul_csparse.iarr",
        ),
        (
            [100, 100],
            None,
            None,
            [43, 23],
            [93, 93],
            True,
            "test_matmul_acontiguous.iarr",
            [100, 100],
            [20, 20],
            [20, 2],
            [12, 42],
            [82, 82],
            True,
            "test_matmul_bcontiguous.iarr",
            [20, 20],
            [8, 11],
            np.float64,
            True,
            None,
        ),
        (
            [100, 100],
            [20, 20],
            [5, 6],
            [15, 15],
            [75, 85],
            True,
            "test_matmul_acontiguous.iarr",
            [100, 100],
            None,
            None,
            [22, 22],
            [92, 32],
            True,
            "test_matmul_bcontiguous.iarr",
            None,
            None,
            np.float32,
            False,
            "test_matmul_csparse.iarr",
        ),
        (
            [1000, 500],
            [200, 200],
            [50, 50],
            [144, 55],
            [964, 465],
            False,
            "test_matmul_asparse.iarr",
            [500],
            [200],
            [90],
            [12],
            [422],
            False,
            None,
            [160],
            [50],
            np.float64,
            False,
            "test_matmul_csparse.iarr",
        ),
        (
            [1000, 1000],
            [300, 300],
            [12, 50],
            [12, 20],
            [320, 300],
            True,
            None,
            [2000],
            [500],
            [100],
            [140],
            [420],
            True,
            None,
            [100],
            [40],
            np.float32,
            True,
            "test_matmul_ccontiguous.iarr",
        ),
    ],
)
def test_matmul_slice(
    ashape,
    achunks,
    ablocks,
    astart,
    astop,
    acontiguous,
    aurlpath,
    bshape,
    bchunks,
    bblocks,
    bstart,
    bstop,
    bcontiguous,
    burlpath,
    cchunks,
    cblocks,
    dtype,
    ccontiguous,
    curlpath,
):
    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)
    ia.remove_urlpath(curlpath)

    acfg = ia.Config(chunks=achunks, blocks=ablocks, contiguous=acontiguous, urlpath=aurlpath)
    a = ia.linspace(ashape, -1, -2, dtype=dtype, cfg=acfg)
    an = ia.iarray2numpy(a)
    aslices = tuple(slice(astart[i], astop[i]) for i in range(len(astart)))
    if len(astart) == 1:
        aslices = aslices[0]
    asl = a[aslices]

    bcfg = ia.Config(chunks=bchunks, blocks=bblocks, contiguous=bcontiguous, urlpath=burlpath)
    b = ia.linspace(bshape, 1, 200, dtype=dtype, cfg=bcfg)
    bn = ia.iarray2numpy(b)
    bslices = tuple(slice(bstart[i], bstop[i]) for i in range(len(bstart)))
    if len(bstart) == 1:
        bslices = bslices[0]
    bsl = b[bslices]

    ccfg = ia.Config(chunks=cchunks, blocks=cblocks, contiguous=ccontiguous, urlpath=curlpath)
    c = ia.matmul(asl, bsl, cfg=ccfg)
    cn = np.matmul(an[aslices], bn[bslices])

    cn_2 = ia.iarray2numpy(c)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(cn, cn_2, rtol=rtol)

    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)
    ia.remove_urlpath(curlpath)


@pytest.mark.parametrize(
    "ashape, bshape, acontiguous, aurlpath, bcontiguous, burlpath",
    [
        ([20, 20], [20, 20], True, None, True, None),
        ([100, 100], [100, 100], False, None, False, None),
        (
            [500, 100],
            [100, 200],
            False,
            "test_matmul_asparse.iarr",
            True,
            "test_matmul_bcontiguous.iarr",
        ),
        (
            [1230, 763],
            [763, 4],
            False,
            "test_matmul_asparse.iarr",
            False,
            "test_matmul_bsparse.iarr",
        ),
        (
            [20, 20],
            [20],
            True,
            "test_matmul_acontiguous.iarr",
            True,
            "test_matmul_bcontiguous.iarr",
        ),
        ([100, 100], [100], True, None, False, "test_matmul_bsparse.iarr"),
        ([500, 100], [100], False, "test_matmul_asparse.iarr", False, None),
        ([1000, 555], [555], True, None, False, None),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matmul_opt(ashape, bshape, dtype, acontiguous, aurlpath, bcontiguous, burlpath):

    params = ia.matmul_params(ashape, bshape, l2_size=1024, chunk_size=16 * 1024)

    achunks, ablocks, bchunks, bblocks = params

    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)

    acfg = ia.Config(chunks=achunks, blocks=ablocks, contiguous=acontiguous, urlpath=aurlpath)
    a = ia.linspace(ashape, -10, 1, dtype=dtype, cfg=acfg)
    an = ia.iarray2numpy(a)

    bcfg = ia.Config(chunks=bchunks, blocks=bblocks,  contiguous=bcontiguous, urlpath=burlpath)
    b = ia.linspace(bshape, -1, 10, dtype=dtype, cfg=bcfg)
    bn = ia.iarray2numpy(b)

    c = ia.matmul(a, b)
    cn_2 = ia.iarray2numpy(c)

    cn = np.matmul(an, bn)

    rtol = 1e-5 if dtype == np.float32 else 1e-12

    np.testing.assert_allclose(cn, cn_2, rtol=rtol)

    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)
