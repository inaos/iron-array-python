# Exercises the Views of actual arrays

import iarray as ia
import numpy as np

# Create an ironArray array
a = ia.arange([10], dtype=np.int8)

# Get a slice (hardcoded view).
# In the future we want to convert this to a regular view.
v = a[:]
print("View slice:", v)
print("data ->", v.data)

# Get a typed view
# v = ia.View(a, dtype=np.bool_)
v = a.astype(np.bool_)
print("View dtype:", v)
print(f"- View dtype: {v.dtype}")
print(f"- Orig dtype: {a.dtype}")
print("data ->", v.data)
