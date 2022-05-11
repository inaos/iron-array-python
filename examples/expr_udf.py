import iarray as ia
from iarray.expr_udf import expr_udf


a = ia.arange([10, 10])
b = ia.arange([10, 10])
print(a.data)
print(b.data)
expr = expr_udf(
    'a[b > 5 and not a < 8 or b > 42]',
    {'a': a, 'b': b},
    debug=1,
)
out = expr.eval()
print(out.data)
