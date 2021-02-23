import pytest
import iarray as ia


@pytest.mark.parametrize(
    "clevel, codec, favor, filters, chunkshape, blockshape, enforce_frame",
    [
        (0, ia.Codecs.ZSTD, ia.Favors.BALANCE, [ia.Filters.SHUFFLE], None, None, False),
        (1, ia.Codecs.BLOSCLZ, ia.Favors.SPEED, [ia.Filters.BITSHUFFLE], [50, 50], [20, 20], True),
        (9, ia.Codecs.ZSTD, ia.Favors.CRATIO, [ia.Filters.SHUFFLE, ia.Filters.DELTA], [50, 50], [20, 20], False),
        (6, ia.Codecs.ZSTD, ia.Favors.BALANCE, [ia.Filters.SHUFFLE], [100, 50], [50, 20], False),
        (0, ia.Codecs.ZSTD, ia.Favors.SPEED, [ia.Filters.SHUFFLE], None, None, False),
    ],
)
def test_global_config(clevel, codec, favor, filters, chunkshape, blockshape, enforce_frame):
    storage = ia.Storage(chunkshape, blockshape, enforce_frame=enforce_frame)
    ia.set_config(clevel=clevel, codec=codec, favor=favor, filters=filters, storage=storage)
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.favor == favor
    assert config.filters == filters
    storage2 = config.storage
    assert storage2.chunkshape == chunkshape
    assert storage2.blockshape == blockshape
    assert storage2.enforce_frame == enforce_frame

    # One can pass storage parameters straight to config() dataclass too
    ia.set_config(
        clevel=clevel,
        codec=codec,
        favor=favor,
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

    # Or, we can set defaults via Config (for better auto-completion)
    cfg = ia.Config(
        clevel=clevel,
        codec=codec,
        favor=favor,
        filters=filters,
        chunkshape=chunkshape,
        blockshape=blockshape,
        enforce_frame=False,
    )
    ia.set_config(cfg)
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    storage2 = ia.Storage()
    assert storage2.chunkshape == chunkshape
    assert storage2.blockshape == blockshape
    assert storage2.enforce_frame == False

    # Or, we can use a mix of Config and keyword args
    cfg = ia.Config(
        clevel=clevel,
        codec=codec,
        blockshape=blockshape,
        enforce_frame=False,
    )
    ia.set_config(cfg, filters=filters, chunkshape=chunkshape)
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.favor == favor
    assert config.filters == filters
    storage2 = ia.Storage()
    assert storage2.chunkshape == chunkshape
    assert storage2.blockshape == blockshape
    assert storage2.enforce_frame == False


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
def test_global_config_dtype(chunkshape, blockshape, shape):
    try:
        storage = ia.Storage(chunkshape, blockshape)
        dtshape = ia.DTShape(shape)
        cfg = ia.set_config(dtshape=dtshape, storage=storage)
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
        cfg = ia.set_config(
            dtshape=dtshape,
            chunkshape=chunkshape,
            blockshape=blockshape,
            enforce_frame=True,
        )
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
def test_config_ctx(clevel, codec, filters, chunkshape, blockshape, plainbuffer):
    try:
        storage = ia.Storage(chunkshape, blockshape, plainbuffer=plainbuffer)
        with ia.config(clevel=clevel, codec=codec, filters=filters, storage=storage) as cfg:
            assert cfg.clevel == clevel
            assert cfg.codec == codec
            assert cfg.filters == filters
            assert cfg.storage.chunkshape == chunkshape
            assert cfg.storage.blockshape == blockshape
            assert cfg.storage.plainbuffer == plainbuffer
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
            assert cfg.storage.chunkshape == chunkshape
            assert cfg.storage.blockshape == blockshape
            assert cfg.storage.plainbuffer == False
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
def test_config_ctx_dtype(chunkshape, blockshape, shape):
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


def test_nested_contexts():
    # Set the default to enable compression
    ia.set_config(clevel=5)
    a = ia.ones(ia.DTShape((100, 100)))
    assert a.cratio > 1

    # Now play with contexts and params in calls
    # Disable compression in contexts
    with ia.config(clevel=0) as cfg:
        a = ia.ones(ia.DTShape((100, 100)), cfg, clevel=0)
        assert a.cratio < 1
        # Enable compression in call
        a = ia.ones(ia.DTShape((100, 100)), cfg, clevel=1)
        assert a.cratio > 1
        # Enable compression in nested context
        with ia.config(clevel=1) as cfg:
            a = ia.ones(ia.DTShape((100, 100)), cfg)
            assert a.cratio > 1
            # Disable compression in double nested context
            with ia.config(clevel=0) as cfg:
                a = ia.ones(ia.DTShape((100, 100)), cfg)
                assert a.cratio < 1

    # Finally, the default should be enabling compression again
    a = ia.ones(ia.DTShape((100, 100)))
    assert a.cratio > 1


@pytest.fixture(scope="function", autouse=True)
def cleanup(request):
    # Make the defaults sane for other tests to come
    return request.addfinalizer(ia.reset_config_defaults)
