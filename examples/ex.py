# This uses the binary code in LLVM .bc file for evaluating expressions.  Only meant for developers, really.

import iarray as ia
import numpy
import numpy as np
from time import time


chunks = [20, 20]
blocks = [10, 10]
acontiguous = True
aurlpath = None
print("Config")
acfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)





