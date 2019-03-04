import iarray as ia
import numpy as np

ia.init()

cfg = ia.Config(eval_flags="iterblock", blocksize=0)
print(str(cfg))
print(repr(cfg))

ctx = ia.Context(cfg)
print(str(ctx))
print(repr(ctx))

dtshape = ia.Dtshape(shape=[10, 20], pshape=[3, 7])
print(str(dtshape))
print(repr(dtshape))
