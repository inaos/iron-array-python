# Exercises the copy() with actual arrays and views of them too.

from time import time
import iarray as ia
import numpy as np
from dataclasses import asdict

# Create an ironArray array
a = ia.linspace(
    -10,
    10,
    int(np.prod((100, 100))),
    shape=(100, 100),
    chunks=[50, 50],
    blocks=[20, 20],
    contiguous=True,
    dtype=np.float64,
)

cframe = a.to_cframe()

b = ia.from_cframe(cframe)

print(b.data)
