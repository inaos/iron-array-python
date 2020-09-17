import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize("shape, chunkshape, blockshape, dtype",
                         [
                             ((1,), (1,), (1,), np.float32),
                             ((1000 * 1000,), (256 * 1024,), (16 * 1024,), np.float32),
                             ((1000 * 1000,), (128 * 1024,), (8 * 1024,), np.float64),
                             ((1, 1), (1, 1), (1, 1), np.float64),
                             ((15 * 1000, 15 * 1000), (256, 512), (64, 128), np.float64),
                             ((15 * 1000, 1112 * 1000), (32, 8 * 1024), (8, 2 * 1024), np.float32),
                             ((15 * 1000, 1112 * 1000), (32, 4 * 1024), (8, 1024), np.float64),
                             ((1, 1, 2), (1, 1, 2), (1, 1, 2), np.float64),
                             ((17 * 1000, 3 * 1000, 300 * 1000), (32, 8, 1024), (8, 4, 512), np.float32),
                             ((17 * 1000, 3 * 1000, 300 * 1000), (32, 4, 1024), (8, 2, 512), np.float64),
                             ((2, 1, 1, 4), (2, 1, 1, 4), (2, 1, 1, 4), np.float64),
                             ((1000, 100, 100, 1000), (64, 8, 8, 64), (32, 4, 4, 32), np.float32),
                             ((1000, 100, 100, 1000), (32, 8, 8, 64), (16, 4, 4, 32), np.float64),
                         ])
def test_partition_advice(shape, chunkshape, blockshape, dtype):
    dtshape = ia.dtshape(shape, dtype)
    # We want to specify max for chunskize, blocksize explicitly, because L2/L3 size is CPU-dependent
    chunkshape_, blockshape_ = ia.partition_advice(dtshape, max_chunksize=1024 * 1024, max_blocksize=64 * 1024)

    assert chunkshape_ == chunkshape
    assert blockshape_ == blockshape
