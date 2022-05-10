import iarray as ia
from iarray.eudf import eudf


args = {'x': ia.ones(shape=[10, 10]), 'y': ia.ones(shape=[10, 10])}
expr = eudf("x + y", args, debug=True)
out = expr.eval()
print(out.data)
