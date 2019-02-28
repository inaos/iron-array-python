import iarray as ia
import numpy as np

# Init iarray
ia.init()

# Create iarray context
cfg = ia.config_new()
ctx = ia.context_new(cfg)

# Define array params
shape = (7, 13)
pshape = (2, 3)
size = int(np.prod(shape))

# Create numpy array
a = np.arange(size, dtype=np.float64).reshape(shape)

# Obtain persistent iarray container from numpy array
filename = b'arange.iarray'
b = ia.numpy2iarray(ctx, a, pshape, filename)
ia.container_free(ctx, b)

# Load iarray container from disk
c = ia.from_file(ctx, filename)

# Get numpy array from iarray container
d = ia.iarray2numpy(ctx, c)

# Assert numpy arrays
np.testing.assert_array_equal(a, d)

# Free data
ia.container_free(ctx, c)
ia.context_free(ctx)

# Destroy iarray
ia.destroy()
