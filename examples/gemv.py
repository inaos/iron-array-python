import iarray as ia
import scipy.io
from time import time
import numpy as np


_ = ia.set_config(clevel=9, codec=ia.Codecs.LZ4, filters=[], btune=False, nthreads=8)

# w = scipy.io.mmread("/Users/aleix11alcacer/Downloads/worms20_10NN/worms20_10NN.mtx")
# wnp = w.toarray()

#
# wia = ia.numpy2iarray(wnp, chunks=(1 * 1024, 1 * 1024), blocks=(64, 64), enforce_frame=True, urlpath="gemv.iarray", mode="w")
wia = ia.load("gemv.iarray")
print(wia.info)

# shape = (8 * 2048, 8 * 2048)
# wia = ia.linspace(shape, 0, 1, chunks=(1024, 1024), blocks=(128, 128))
# wia[:250, :] = 0

w = wia.data

bia = ia.linspace((wia.shape[1],), 0, 1, chunks=(wia.chunks[1],), blocks=(wia.blocks[1],))

# b[256:512] = 0
t0 = time()
c = w.dot(bia.data)
t1 = time()
print(f"scipy: {t1 - t0:.5f} s")

t0 = time()
cia = ia.gemv(wia, bia, chunks=(wia.chunks[0],), blocks=(wia.blocks[0],))
t1 = time()

print(f"iarray (gemv): {t1 - t0:.5f} s")

t0 = time()
cia2 = ia.matmul(wia, bia, chunks=(wia.chunks[0],), blocks=(wia.blocks[0],))
t1 = time()

print(f"iarray: {t1 - t0:.5f} s")

np.testing.assert_allclose(cia.data, c, rtol=1e-6)
