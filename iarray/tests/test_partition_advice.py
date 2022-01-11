import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype",
    [
        ((1,), (1,), (1,), np.float32),
        ((1000 * 1000,), (64 * 1024,), (16 * 1024,), np.float32),
        ((1000 * 1000,), (128 * 1024,), (8 * 1024,), np.float64),
        ((1, 1), (1, 1), (1, 1), np.bool_),
        ((15 * 1000, 15 * 1000), (256, 512), (64, 128), np.int64),
        ((15 * 1000, 1112 * 1000), (32, 8 * 1024), (8, 2 * 1024), np.int32),
        ((15 * 1000, 1112 * 1000), (32, 4 * 1024), (8, 1024), np.uint64),
        ((1, 1, 2), (1, 1, 2), (1, 1, 2), np.float64),
        ((17 * 1000, 3 * 1000, 300 * 1000), (32, 8, 1024), (8, 4, 512), np.uint32),
        ((17 * 1000, 3 * 1000, 300 * 1000), (64, 8, 1024), (16, 4, 512), np.int16),
        ((2, 1, 1, 4), (2, 1, 1, 4), (2, 1, 1, 4), np.uint16),
        ((1000, 100, 100, 1000), (64, 8, 16, 128), (32, 4, 8, 64), np.bool_),
        ((1000, 100, 100, 1000), (64, 8, 16, 128), (32, 4, 8, 64), np.int8),
        ((2, 3, 4, 5, 6, 7, 8, 9), (2, 2, 4, 4, 4, 4, 8, 8), (1, 2, 4, 4, 4, 4, 8, 8), np.uint8),
    ],
)
def test_partition_advice(shape, chunks, blocks, dtype):
    # We want to specify max for chunskize, blocksize explicitly, because L2/L3 size is CPU-dependent

    ia.set_config_defaults(dtype=dtype)
    chunks_, blocks_ = ia.partition_advice(
        shape, max_chunksize=1024 * 1024, max_blocksize=64 * 1024
    )

    assert chunks_ == chunks
    assert blocks_ == blocks
