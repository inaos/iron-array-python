from ast_decompiler import decompile

import iarray as ia
from iarray import udf
from iarray.eudf import eudf


tree = eudf("x + y", {'x': ia.ones(shape=[10, 10]), 'y': ia.ones(shape=[10, 10])})
print(decompile(tree))
