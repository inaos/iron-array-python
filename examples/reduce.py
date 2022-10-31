# matmul comparison against numpy.

import iarray as ia
import numpy as np

shape = [5, 5]
chunks = [5, 5]
blocks = [2, 2]

axis = (0, 1)

dtype = np.float64
urlpath1 = "example_linspace.iarr"
urlpath2 = "example_reduce.iarr"

ia.remove_urlpath(urlpath1)
ia.remove_urlpath(urlpath2)

a = ia.arange(
    int(np.prod(shape)),
    shape=shape,
    chunks=chunks,
    blocks=blocks,
    dtype=dtype,
    urlpath=urlpath1,
    btune=False,
    mode="w",
)
slices = tuple([slice(np.random.randint(1, s)) for s in a.shape])
a[slices] = np.nan
print(slices)
a_data = a.data
cn2 = np.nanmean(a_data, axis=axis)
cn = ia.nanmean(a, urlpath=urlpath2, axis=axis)

print(a_data)

# np.testing.assert_allclose(cn.data, cn2)
print("Matrix reduction is working!")

ia.remove_urlpath(urlpath1)
ia.remove_urlpath(urlpath2)

# a = np.random.random_sample(np.prod(shape)).reshape(shape)
# a[...] = np.nan

# a1 = np.nanmean(np.nanmean(a, axis=1), axis=0)
# a2 = np.nanmean(a, axis=(1, 0))
# np.testing.assert_allclose(a1, a2)

print(np.median(a_data, axis=0))
