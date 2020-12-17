import iarray as ia

ia_precip = ia.load("precip-3m.iarr")

precip1 = ia_precip[0]
precip2 = ia_precip[1]
precip3 = ia_precip[2]

@profile
def iarray_eval_disk(expr):
    with ia.config(storage=ia.Storage(filename="mean-3m.iarr")) as cfg:
        expr_val = expr.eval(cfg=cfg)
    return expr_val

mean_expr = (precip1 + precip2 + precip3) / 3
mean_disk = iarray_eval_disk(mean_expr)

@profile
def iarray_trans_disk(expr):
    with ia.config(storage=ia.Storage(filename="trans-3m.iarr")) as cfg:
        expr_val = expr.eval(cfg=cfg)
    return expr_val

trans_expr = ia.tan(precip1) * (ia.sin(precip1) * ia.sin(precip2) + ia.cos(precip2)) + ia.sqrt(precip3) * 2
trans_val = iarray_trans_disk(trans_expr)
