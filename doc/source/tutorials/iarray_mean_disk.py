import iarray as ia

ia_precip = ia.load("precip-3m.iarr")

precip1 = ia_precip[0]
precip2 = ia_precip[1]
precip3 = ia_precip[2]

mean_expr = (precip1 + precip2 + precip3) / 3

@profile
def iarray_eval_disk(mean_expr):
    with ia.config(storage=ia.Storage(filename="mean-3m.iarr")) as cfg:
        mean_disk = mean_expr.eval(cfg=cfg)
    return mean_disk

mean_disk = iarray_eval_disk(mean_expr)
