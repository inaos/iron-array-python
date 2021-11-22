import pytest
import iarray as ia


@pytest.mark.parametrize(
    "clevel, codec, filters, chunks, blocks, contiguous, urlpath",
    [
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None, False, None),
        (1, ia.Codec.BLOSCLZ, [ia.Filter.SHUFFLE], [50, 50], [20, 20], True, None),
        (
            9,
            ia.Codec.ZSTD,
            [ia.Filter.SHUFFLE, ia.Filter.DELTA],
            [50, 50],
            [20, 20],
            False,
            b"test_config_params_sparse.iarr",
        ),
        (
            6,
            ia.Codec.ZSTD,
            [ia.Filter.SHUFFLE],
            [100, 50],
            [50, 20],
            True,
            b"test_config_params_contiguous.iarr",
        ),
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None, False, None),
    ],
)
def test_global_config(clevel, codec, filters, chunks, blocks, contiguous, urlpath):
    store = ia.Store(chunks, blocks, contiguous=contiguous, urlpath=urlpath)
    ia.set_config(clevel=clevel, codec=codec, filters=filters, store=store, btune=False)
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    store2 = config.store
    assert store2.chunks == chunks
    assert store2.blocks == blocks
    assert store2.contiguous == contiguous
    assert store2.urlpath == urlpath

    # One can pass store parameters straight to config() dataclass too
    ia.set_config(
        clevel=clevel,
        codec=codec,
        filters=filters,
        chunks=chunks,
        blocks=blocks,
        contiguous=False,
        urlpath=None,
        btune=False,
    )
    config = ia.get_config()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    store2 = ia.Store()
    assert store2.chunks == chunks
    assert store2.blocks == blocks
    assert store2.contiguous == False
    assert store2.urlpath == None

    # Or, we can set defaults via Config (for better auto-completion)
    cfg = ia.Config(
        clevel=clevel,
        codec=codec,
        filters=filters,
        chunks=chunks,
        blocks=blocks,
        contiguous=False,
        urlpath=None,
        btune=False,
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
    assert store2.urlpath == None

    # Or, we can use a mix of Config and keyword args
    cfg = ia.Config(clevel=clevel, codec=codec, blocks=blocks, contiguous=False, urlpath=urlpath, btune=False)
    ia.set_config(cfg, filters=filters, chunks=chunks)
    config = ia.get_config()

    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    store2 = ia.Store()
    assert store2.chunks == chunks
    assert store2.blocks == blocks
    assert store2.contiguous == False
    assert store2.urlpath == urlpath


@pytest.mark.parametrize(
    "favor, chunks, blocks",
    [
        (ia.Favor.BALANCE, None, None),
        (ia.Favor.SPEED, [50, 50], [20, 20]),
        (ia.Favor.CRATIO, [50, 50], [20, 20]),
    ],
)
def test_global_favor(favor, chunks, blocks):
    store = ia.Store(chunks, blocks)
    ia.set_config(favor=favor, store=store)
    config = ia.get_config()
    assert config.favor == favor
    assert config.btune == True
    assert config.store.chunks == chunks
    assert config.store.blocks == blocks


@pytest.mark.parametrize(
    "favor",
    [
        (ia.Favor.BALANCE, ia.Favor.SPEED, ia.Favor.CRATIO),
    ],
)
def test_favor_nobtune(favor):
    with pytest.raises(ValueError):
        ia.set_config(favor=favor, btune=False)


@pytest.mark.parametrize(
    "clevel, codec, filters",
    [
        (1, None, None),
        (None, ia.Codec.ZSTD, None),
        (None, None, [ia.Filter.SHUFFLE]),
        (1, ia.Codec.ZSTD, [ia.Filter.SHUFFLE]),
    ],
)
def test_btune_incompat(clevel, codec, filters):
    with pytest.raises(ValueError):
        ia.set_config(clevel=clevel, codec=codec, filters=filters, btune=True)

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
        assert shape == ()


@pytest.mark.parametrize(
    "clevel, codec, filters, chunks, blocks",
    [
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None),
        (1, ia.Codec.BLOSCLZ, [ia.Filter.BITSHUFFLE], [50, 50], [20, 20]),
        (9, ia.Codec.ZSTD, [ia.Filter.SHUFFLE, ia.Filter.DELTA], [50, 50], [20, 20]),
        (6, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], [100, 50], [50, 20]),
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None),
    ],
)
def test_config_ctx(clevel, codec, filters, chunks, blocks):
    try:
        store = ia.Store(chunks, blocks)
        with ia.config(clevel=clevel, codec=codec, filters=filters, store=store, btune=False) as cfg:
            assert cfg.clevel == clevel
            assert cfg.codec == codec
            assert cfg.btune == False
            assert cfg.filters == filters
            assert cfg.store.chunks == chunks
            assert cfg.store.blocks == blocks
    except ValueError:
        assert chunks is not None

    # One can pass store parameters straight to config() dataclass too
    try:
        with ia.config(
            clevel=clevel,
            codec=codec,
            filters=filters,
            chunks=chunks,
            blocks=blocks,
            btune=False,
        ) as cfg:
            assert cfg.clevel == clevel
            assert cfg.codec == codec
            assert cfg.btune == False
            assert cfg.filters == filters
            assert cfg.store.chunks == chunks
            assert cfg.store.blocks == blocks
    except ValueError:
        assert chunks is not None


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
            assert store2.contiguous is True
    except ValueError:
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


def test_default_params():
    urlpath = "arr.iarr"
    ia.remove_urlpath(urlpath)

    cfg = ia.get_config()
    a = ia.linspace([10], start=0, stop=1, urlpath=urlpath, contiguous=False)
    cfg2 = ia.Config()

    assert cfg.contiguous == cfg2.contiguous
    assert cfg.urlpath == cfg2.urlpath
    ia.remove_urlpath(urlpath)


@pytest.fixture(scope="function", autouse=True)
def cleanup(request):
    # Make the defaults sane for other tests to come
    return request.addfinalizer(ia.reset_config_defaults)
