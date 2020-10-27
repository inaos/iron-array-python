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

    # One can pass storage parameters straight to config() dataclass too
    ia.set_config(
        clevel=clevel,
        codec=codec,
        filters=filters,
        chunkshape=chunkshape,
        blockshape=blockshape,
        enforce_frame=False,
    )
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    storage2 = ia.Storage()
    assert storage2.chunkshape == chunkshape
    assert storage2.blockshape == blockshape
    assert storage2.enforce_frame == False


@pytest.mark.parametrize(
    "clevel, codec, filters, chunkshape, blockshape, plainbuffer",
    [
        (0, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE], None, None, False),
        (1, ia.Codecs.BLOSCLZ, [ia.Filters.BITSHUFFLE], [50, 50], [20, 20], True),
        (9, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE, ia.Filters.DELTA], [50, 50], [20, 20], False),
        (6, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE], [100, 50], [50, 20], False),
        (0, ia.Codecs.ZSTD, [ia.Filters.SHUFFLE], None, None, False),
    ],
)
def test_cparams_ctx(clevel, codec, filters, chunkshape, blockshape, plainbuffer):
    try:
        storage = ia.Storage(chunkshape, blockshape, plainbuffer=plainbuffer)
        with ia.config(clevel=clevel, codec=codec, filters=filters, storage=storage) as cfg:
            assert cfg.clevel == clevel
            assert cfg.codec == codec
            assert cfg.filters == filters
            storage2 = ia.Storage()
            assert storage2.chunkshape == chunkshape
            assert storage2.blockshape == blockshape
            assert storage2.plainbuffer == plainbuffer
    except ValueError:
        # chunkshape cannot be set when a plainbuffer is used
        assert plainbuffer and chunkshape is not None

    # One can pass storage parameters straight to config() dataclass too
    try:
        with ia.config(
            clevel=clevel,
            codec=codec,
            filters=filters,
            chunkshape=chunkshape,
            blockshape=blockshape,
            plainbuffer=False,
        ) as cfg:
            assert cfg.clevel == clevel
            assert cfg.codec == codec
            assert cfg.filters == filters
            storage2 = ia.Storage()
            assert storage2.chunkshape == chunkshape
            assert storage2.blockshape == blockshape
            assert storage2.plainbuffer == False
    except ValueError:
        # chunkshape cannot be set when a plainbuffer is used
        assert plainbuffer and chunkshape is not None


@pytest.mark.parametrize(
    "chunkshape, blockshape, shape",
    [
        (None, None, (100, 100)),
        ((50, 50), (20, 20), (100, 100)),
        ((50, 50), (20, 20), [10, 100]),
        ((100, 50), (50, 20), [100, 10]),
        (None, None, ()),
    ],
)
def test_cparams_ctx_dtype(chunkshape, blockshape, shape):
    try:
        storage = ia.Storage(chunkshape, blockshape)
        dtshape = ia.DTShape(shape)
        with ia.config(dtshape=dtshape, storage=storage) as cfg:
            storage2 = cfg.storage
            if chunkshape is not None:
                assert storage2.chunkshape == chunkshape
                assert storage2.blockshape == blockshape
            else:
                # automatic partitioning
                assert storage2.chunkshape <= shape
                assert storage2.blockshape <= storage2.chunkshape
    except ValueError:
        # chunkshape cannot be set when a plainbuffer is used
        assert shape == ()

    # One can pass storage parameters straight to config() dataclass too
    try:
        dtshape = ia.DTShape(shape)
        with ia.config(
            dtshape=dtshape,
            chunkshape=chunkshape,
            blockshape=blockshape,
            enforce_frame=True,
        ) as cfg:
            storage2 = cfg.storage
            if chunkshape is not None:
                assert storage2.chunkshape == chunkshape
                assert storage2.blockshape == blockshape
            else:
                # automatic partitioning
                assert storage2.chunkshape <= shape
                assert storage2.blockshape <= storage2.chunkshape
            assert storage2.enforce_frame == True
    except ValueError:
        # chunkshape cannot be set when a plainbuffer is used
        assert shape == ()
