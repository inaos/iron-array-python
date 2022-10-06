# Exercises the copy() with actual arrays and views of them too.

from time import time
import iarray as ia
import numpy as np
from dataclasses import asdict

# Create an ironArray array
a = ia.arange([10, 10], chunks=[4, 5], blocks=[2, 2], contiguous=True, dtype=np.int64)

split = a.split()
print(a.split())


b = ia.concatenate(a.shape, split)

print(b.slice_chunk_index((4, 10), [0, 1]).data)
