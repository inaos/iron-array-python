from time import time

t0 = time()
import iarray as ia
import zarr

zarr.save_array()
t = time() - t0
print("iarray import time ->", round(t, 3))

# ia_precip = ia.open("precip-3m.iarr")
# chunkshape = ia_precip.chunkshape[1:]
# blockshape = ia_precip.blockshape[1:]

# print("-->", ia_precip.info)

# precip1 = ia_precip[0].copy(chunkshape=chunkshape, blockshape=blockshape)
# precip2 = ia_precip[1].copy(chunkshape=chunkshape, blockshape=blockshape)
# precip3 = ia_precip[2].copy(chunkshape=chunkshape, blockshape=blockshape)
# print("1 -->", precip1.info)

# precip1 = ia_precip[0].copy()
# precip2 = ia_precip[1].copy()
# precip3 = ia_precip[2].copy()
# print("1 -->", precip1.info)

# ia.save("precip1.iarr", precip1)
# ia.save("precip2.iarr", precip2)
# ia.save("precip3.iarr", precip3)

# @profile
def iarray_open_data():
    precip1 = ia.open("precip1.iarr")
    precip2 = ia.open("precip2.iarr")
    precip3 = ia.open("precip3.iarr")
    return precip1, precip2, precip3


t0 = time()
precip1, precip2, precip3 = iarray_open_data()
t = time() - t0
print("open time ->", round(t, 3))

print("2 -->", precip1.info)
chunkshape = precip1.chunkshape
blockshape = precip1.blockshape

storage = ia.Storage(chunkshape=precip1.chunkshape, blockshape=precip1.blockshape)
# cfg = ia.Config(nthreads=14, storage=storage, clevel=1, codec=ia.Codecs.ZSTD, filters=[ia.Filters.BITSHUFFLE])
cfg = ia.Config(storage=storage)


@profile
def iarray_mean_disk(expr):
    with ia.config(urlpath="mean-3m.iarr", cfg=cfg) as cfg2:
        expr_val = expr.eval(cfg=cfg2)
    return expr_val


t0 = time()
mean_expr = (precip1 + precip2 + precip3) / 3
t = time() - t0
print("mean expr time ->", round(t, 3))

t0 = time()
mean_disk = iarray_mean_disk(mean_expr)
t = time() - t0
print("mean eval time ->", round(t, 3))
mean_disk.info


@profile
def iarray_trans_disk(expr):
    with ia.config(urlpath="trans-3m.iarr", cfg=cfg) as cfg2:
        expr_val = expr.eval(cfg=cfg2)
    return expr_val


trans_expr = (
    ia.tan(precip1) * (ia.sin(precip1) * ia.sin(precip2) + ia.cos(precip2)) + ia.sqrt(precip3) * 2
)
t0 = time()
trans_val = iarray_trans_disk(trans_expr)
t = time() - t0
print("trans eval time ->", round(t, 3))
