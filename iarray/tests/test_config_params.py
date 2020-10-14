import pytest
import iarray as ia


@pytest.mark.parametrize(
    "clevel, codec, filters, chunkshape, blockshape, enforce_frame",
    [
        (0, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE], None, None, False),
        (1, ia.Codecs.BLOSCLZ, [ia.Filters.BITSHUFFLE], [50, 50], [20, 20], True),
        (9, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE, ia.Filters.DELTA], [50, 50], [20, 20], False),
        (6, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE], [100, 50], [50, 20], False),
        (0, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE], None, None, False),
    ],
)
def test_cparams(clevel, codec, filters, chunkshape, blockshape, enforce_frame):
    storage = ia.Storage(chunkshape, blockshape, enforce_frame=enforce_frame)
    ia.set_config(clevel=clevel, codec=codec, filters=filters, storage=storage)
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    storage2 = ia.Storage()
    assert storage2.chunkshape == chunkshape
    assert storage2.blockshape == blockshape
    assert storage2.enforce_frame == enforce_frame


@pytest.mark.parametrize(
    "clevel, codec, filters, chunkshape, blockshape, enforce_frame",
    [
        (0, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE], None, None, False),
        (1, ia.Codecs.BLOSCLZ, [ia.Filters.BITSHUFFLE], [50, 50], [20, 20], True),
        (9, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE, ia.Filters.DELTA], [50, 50], [20, 20], False),
        (6, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE], [100, 50], [50, 20], False),
        (0, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE], None, None, False),
    ],
)
def test_cparams_ctx(clevel, codec, filters, chunkshape, blockshape, enforce_frame):
    storage = ia.Storage(chunkshape, blockshape, enforce_frame=enforce_frame)
    with ia.config(clevel=clevel, codec=codec, filters=filters, storage=storage):
        config = ia.get_config()
        assert config.clevel == clevel
        assert config.codec == codec
        assert config.filters == filters
        storage2 = ia.Storage()
        assert storage2.chunkshape == chunkshape
        assert storage2.blockshape == blockshape
        assert storage2.enforce_frame == enforce_frame
