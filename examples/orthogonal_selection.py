# Examples on getting slices

import iarray as ia
import numpy as np


dtype = np.float32

ia.set_config_defaults(dtype=dtype)

a = ia.full((10, 10), 3.14, chunks=(4, 6), blocks=(3, 3))
print(a.info)

a[5:10, 7:10] = np.ones(5 * 3, dtype=dtype)
a[1:3] = np.zeros(10 * 2, dtype=dtype)
a[6:9, 2:4] = -1.2

selection = [slice(2, 8), -6]
a.set_orthogonal_selection(selection, -0.01)
print(a.data)

b = a.get_orthogonal_selection(selection)
print(b)

c = ia.arange(shape=[3, 5], dtype=np.int64)
print(c.oindex[[0, 2], :])
print(c.oindex[:, [1, 3]])
print(c.oindex[[0, 2], [1, 3]])
c.oindex[[0, 2], [1, 3]] = [[-1, -2], [-3, -4]]
print(c.data)
