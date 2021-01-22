import iarray as ia

ia_precip = ia.open("precip-3m.iarr")
chunkshape = ia_precip.chunkshape[1:]
blockshape = ia_precip.blockshape[1:]

print("-->", ia_precip.info)

#precip1 = ia_precip[0].copy(chunkshape=chunkshape, blockshape=blockshape)
#precip2 = ia_precip[1].copy(chunkshape=chunkshape, blockshape=blockshape)
#precip3 = ia_precip[2].copy(chunkshape=chunkshape, blockshape=blockshape)
#print("1 -->", precip1.info)

#precip1 = ia_precip[0].copy()
#precip2 = ia_precip[1].copy()
#precip3 = ia_precip[2].copy()
#print("1 -->", precip1.info)

#ia.save(precip1, "precip1.iarr")
#ia.save(precip2, "precip2.iarr")
#ia.save(precip3, "precip3.iarr")

precip1 = ia.open("precip1.iarr")
precip2 = ia.open("precip2.iarr")
precip3 = ia.open("precip3.iarr")
print("2 -->", precip1.info)
chunkshape = precip1.chunkshape
blockshape = precip1.blockshape


@profile
def iarray_eval_disk(expr):
    with ia.config(storage=ia.Storage(filename="mean-3m.iarr", chunkshape=chunkshape, blockshape=blockshape), nthreads=0) as cfg:
    #with ia.config() as cfg:
        expr_val = expr.eval(cfg=cfg)
    return expr_val

mean_expr = (precip1 + precip2 + precip3) / 3
mean_disk = iarray_eval_disk(mean_expr)
mean_disk.info

@profile
def iarray_trans_disk(expr):
    with ia.config(storage=ia.Storage(filename="trans-3m.iarr", chunkshape=chunkshape, blockshape=blockshape), nthreads=0) as cfg:
    #with ia.config() as cfg:
        expr_val = expr.eval(cfg=cfg)
    return expr_val

trans_expr = ia.tan(precip1) * (ia.sin(precip1) * ia.sin(precip2) + ia.cos(precip2)) + ia.sqrt(precip3) * 2
trans_val = iarray_trans_disk(trans_expr)
