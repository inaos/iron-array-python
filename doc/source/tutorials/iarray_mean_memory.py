import iarray as ia

#ia_precip = ia.open("precip-3m.iarr")
#precip1 = ia_precip[0].copy()
#precip2 = ia_precip[1].copy()
#precip3 = ia_precip[2].copy()

precip1 = ia.load("precip1.iarr")
precip2 = ia.load("precip2.iarr")
precip3 = ia.load("precip3.iarr")
precip1.info

@profile
def iarray_mean_memory(expr):
    expr_val = expr.eval(nthreads=0, enforce_frame=False)
    return expr_val

mean_expr = (precip1 + precip2 + precip3) / 3
mean_val = iarray_mean_memory(mean_expr)

@profile
def iarray_trans_memory(expr):
    expr_val = expr.eval(nthreads=0, enforce_frame=False)
    return expr_val

trans_expr = ia.tan(precip1) * (ia.sin(precip1) * ia.sin(precip2) + ia.cos(precip2)) + ia.sqrt(precip3) * 2
trans_val = iarray_trans_memory(trans_expr)
