import pytest
import iarray as ia


@pytest.mark.parametrize(
    "clevel, codec, filters, chunks, blocks, contiguous",
    [
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None, False),
        (1, ia.Codec.BLOSCLZ, [ia.Filter.SHUFFLE], [50, 50], [20, 20], True),
        (9, ia.Codec.ZSTD, [ia.Filter.SHUFFLE, ia.Filter.DELTA], [50, 50], [20, 20], False),
        (6, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], [100, 50], [50, 20], False),
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None, False),
    ],
)
def test_global_config(clevel, codec, filters, chunks, blocks, contiguous):
    store = ia.Store(chunks, blocks, contiguous=contiguous)
    ia.set_config(clevel=clevel, codec=codec, filters=filters, store=store)
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    store2 = config.store
    assert store2.chunks == chunks
    assert store2.blocks == blocks
    assert store2.contiguous == contiguous

    # One can pass store parameters straight to config() dataclass too
    ia.set_config(
        clevel=clevel,
        codec=codec,
        filters=filters,
        chunks=chunks,
        blocks=blocks,
        contiguous=False,
    )
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    store2 = ia.Store()
    assert store2.chunks == chunks
    assert store2.blocks == blocks
    assert store2.contiguous == False

    # Or, we can set defaults via Config (for better auto-completion)
    cfg = ia.Config(
        clevel=clevel,
        codec=codec,
        filters=filters,
        chunks=chunks,
        blocks=blocks,
        contiguous=False,
    )
    ia.set_config(cfg)
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    store2 = ia.Store()
    assert store2.chunks == chunks
    assert store2.blocks == blocks
    assert store2.contiguous == False

    # Or, we can use a mix of Config and keyword args
    cfg = ia.Config(
        clevel=clevel,
        codec=codec,
        blocks=blocks,
        contiguous=False,
    )
    ia.set_config(cfg, filters=filters, chunks=chunks)
    config = ia.get_config()

    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    store2 = ia.Store()
    assert store2.chunks == chunks
    assert store2.blocks == blocks
    assert store2.contiguous == False


@pytest.mark.parametrize(
    "favor, filters, chunks, blocks",
    [
        (ia.Favor.BALANCE, [ia.Filter.BITSHUFFLE], None, None),
        (ia.Favor.SPEED, [ia.Filter.SHUFFLE], [50, 50], [20, 20]),
        (ia.Favor.CRATIO, [ia.Filter.BITSHUFFLE], [50, 50], [20, 20]),
    ],
)
def test_global_favor(favor, filters, chunks, blocks):
    store = ia.Store(chunks, blocks)
    ia.set_config(favor=favor, filters=filters, store=store)
    config = ia.get_config()
    assert config.favor == favor
    assert config.filters == filters
    assert config.store.chunks == chunks
    assert config.store.blocks == blocks


@pytest.mark.parametrize(
    "chunks, blocks, shape",
    [
        (None, None, (100, 100)),
        ((50, 50), (20, 20), (100, 100)),
        ((50, 50), (20, 20), [10, 100]),
        ((100, 50), (50, 20), [100, 10]),
        (None, None, ()),
    ],
)
def test_global_config_dtype(chunks, blocks, shape):
    try:
        store = ia.Store(chunks, blocks)
        cfg = ia.set_config(shape=shape, store=store)
        store2 = cfg.store

        if chunks is not None:
            assert store2.chunks == chunks
            assert store2.blocks == blocks
        else:
            # automatic partitioning
            assert store2.chunks <= shape
            assert store2.blocks <= store2.chunks
    except ValueError:
        # chunks cannot be set when a plainbuffer is used
        assert shape == ()

    # One can pass store parameters straight to config() dataclass too
    try:
        cfg = ia.set_config(
            shape=shape,
            chunks=chunks,
            blocks=blocks,
            contiguous=True,
        )
        store2 = cfg.store

        if chunks is not None:
            assert store2.chunks == chunks
            assert store2.blocks == blocks
        else:
            # automatic partitioning
            assert store2.chunks <= shape
            assert store2.blocks <= store2.chunks
        assert store2.contiguous == True
    except ValueError:
        # chunks cannot be set when a plainbuffer is used
        assert shape == ()


@pytest.mark.parametrize(
    "clevel, codec, filters, chunks, blocks, plainbuffer",
    [
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None, False),
        (1, ia.Codec.BLOSCLZ, [ia.Filter.BITSHUFFLE], [50, 50], [20, 20], True),
        (9, ia.Codec.ZSTD, [ia.Filter.SHUFFLE, ia.Filter.DELTA], [50, 50], [20, 20], False),
        (6, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], [100, 50], [50, 20], False),
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None, False),
    ],
)
def test_config_ctx(clevel, codec, filters, chunks, blocks, plainbuffer):
    try:
        store = ia.Store(chunks, blocks, plainbuffer=plainbuffer)
        with ia.config(clevel=clevel, codec=codec, filters=filters, store=store) as cfg:
            assert cfg.clevel == clevel
            assert cfg.codec == codec
            assert cfg.filters == filters
            assert cfg.store.chunks == chunks
            assert cfg.store.blocks == blocks
            assert cfg.store.plainbuffer == plainbuffer
    except ValueError:
        # chunks cannot be set when a plainbuffer is used
        assert plainbuffer and chunks is not None

    # One can pass store parameters straight to config() dataclass too
    try:
        with ia.config(
            clevel=clevel,
            codec=codec,
            filters=filters,
            chunks=chunks,
            blocks=blocks,
            plainbuffer=False,
        ) as cfg:
            assert cfg.clevel == clevel
            assert cfg.codec == codec
            assert cfg.filters == filters
            assert cfg.store.chunks == chunks
            assert cfg.store.blocks == blocks
            assert cfg.store.plainbuffer == False
    except ValueError:
        # chunks cannot be set when a plainbuffer is used
        assert plainbuffer and chunks is not None


@pytest.mark.parametrize(
    "chunks, blocks, shape",
    [
        (None, None, (100, 100)),
        ((50, 50), (20, 20), (100, 100)),
        ((50, 50), (20, 20), [10, 100]),
        ((100, 50), (50, 20), [100, 10]),
        (None, None, ()),
    ],
)
def test_config_ctx_dtype(chunks, blocks, shape):
    try:
        store = ia.Store(chunks, blocks)
        with ia.config(shape=shape, store=store) as cfg:
            store2 = cfg.store
            if chunks is not None:
                assert store2.chunks == chunks
                assert store2.blocks == blocks
            else:
                # automatic partitioning
                assert store2.chunks <= shape
                assert store2.blocks <= store2.chunks
    except ValueError:
        # chunks cannot be set when a plainbuffer is used
        assert shape == ()

    # One can pass store parameters straight to config() dataclass too
    try:
        with ia.config(
            shape=shape,
            chunks=chunks,
            blocks=blocks,
            contiguous=True,
        ) as cfg:
            store2 = cfg.store
            if chunks is not None:
                assert store2.chunks == chunks
                assert store2.blocks == blocks
            else:
                # automatic partitioning
                assert store2.chunks <= shape
                assert store2.blocks <= store2.chunks
            assert store2.contiguous == True
    except ValueError:
        # chunks cannot be set when a plainbuffer is used
        assert shape == ()


def test_nested_contexts():
    # Set the default to enable compression
    ia.set_config(clevel=5, btune=False)
    a = ia.ones((100, 100))
    b = a.data

    assert a.cratio > 1

    # Now play with contexts and params in calls
    # Disable compression in contexts
    with ia.config(clevel=0):
        a = ia.numpy2iarray(b)
        assert a.cratio < 1
        # Enable compression in call
        a = ia.numpy2iarray(b, clevel=1)
        assert a.cratio > 1
        # Enable compression in nested context
        with ia.config(clevel=1):
            a = ia.numpy2iarray(b)
            assert a.cratio > 1
            # Disable compression in double nested context
            with ia.config(clevel=0):
                a = ia.numpy2iarray(b)
                assert a.cratio < 1

    # Finally, the default should be enabling compression again
    a = ia.ones((100, 100))
    assert a.cratio > 1


@pytest.fixture(scope="function", autouse=True)
def cleanup(request):
    # Make the defaults sane for other tests to come
    return request.addfinalizer(ia.reset_config_defaults)
