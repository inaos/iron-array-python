import iarray as ia
from iarray.expr_udf import expr_udf


args = {'x': ia.ones(shape=[10, 10]), 'y': ia.ones(shape=[10, 10])}
expr = expr_udf("x + y", args, debug=1)
out = expr.eval()
print(out.data)
