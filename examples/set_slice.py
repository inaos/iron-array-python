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

print(a.data)
