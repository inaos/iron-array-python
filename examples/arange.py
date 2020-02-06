import iarray as ia
import numpy as np
import os.path


filename = 'arange.iarray'
shape = (7, 13)
pshape = (2, 3)
dtype = np.float32

print("Start linspace")
a_ = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10)
sl = tuple([slice(0, s - 1) for s in shape])
a = a_[sl]

print("Start copy")
b = a.copy()
print("End copy")
c = a.copy(view=True)
bn = ia.iarray2numpy(b)
cn = ia.iarray2numpy(c)
