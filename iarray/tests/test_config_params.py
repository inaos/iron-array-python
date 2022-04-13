import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "clevel, codec, filters, chunks, blocks, contiguous, urlpath, meta",
    [
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None, False, None, None),
        (1, ia.Codec.BLOSCLZ, [ia.Filter.SHUFFLE], [50, 50], [20, 20], True, None, None),
        (
            9,
            ia.Codec.ZSTD,
            [ia.Filter.SHUFFLE, ia.Filter.DELTA],
            [50, 50],
            [20, 20],
            False,
            b"test_config_params_sparse.iarr",
            None,
        ),
        (
            6,
            ia.Codec.ZSTD,
            [ia.Filter.SHUFFLE],
            [100, 50],
            [50, 20],
            True,
            b"test_config_params_contiguous.iarr",
            None,
        ),
        (0, ia.Codec.ZSTD, [ia.Filter.SHUFFLE], None, None, False, None, None),
        (0, ia.Codec.ZFP_FIXED_ACCURACY, [ia.Filter.SHUFFLE], None, None, False, None, -2),
    ],
)
def test_global_config(clevel, codec, filters, chunks, blocks, contiguous, urlpath, meta):
    ia.set_config_defaults(clevel=clevel, codec=codec, filters=filters, chunks=chunks, blocks=blocks,
                           contiguous=contiguous, urlpath=urlpath, btune=False, meta=meta)
    config = ia.get_config_defaults()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    assert config.chunks == chunks
    assert config.blocks == blocks
    assert config.contiguous == contiguous
    assert config.urlpath == urlpath
    if meta is None:
        assert config.meta == 0
    else:
        assert config.meta == meta

    # One can pass store parameters straight to config() dataclass too
    ia.set_config_defaults(
        clevel=clevel,
        codec=codec,
        filters=filters,
        chunks=chunks,
        blocks=blocks,
        contiguous=False,
        urlpath=None,
        btune=False,
        meta=meta,
    )
    config = ia.get_config_defaults()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    assert config.chunks == chunks
    assert config.blocks == blocks
    assert config.contiguous is False
    assert config.urlpath is None
    if meta is None:
        assert config.meta == 0
    else:
        assert config.meta == meta

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
        meta=meta,
    )
    ia.set_config_defaults(cfg)
    config = ia.get_config_defaults()
    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    assert config.chunks == chunks
    assert config.blocks == blocks
    assert config.contiguous is False
    assert config.urlpath is None
    if meta is None:
        assert config.meta == 0
    else:
        assert config.meta == meta

    # Or, we can use a mix of Config and keyword args
    cfg = ia.Config(
        clevel=clevel, codec=codec, blocks=blocks, contiguous=False, urlpath=urlpath, btune=False
    )
    ia.set_config_defaults(cfg, filters=filters, chunks=chunks, meta=meta)
    config = ia.get_config_defaults()

    assert config.clevel == clevel
    assert config.codec == codec
    assert config.filters == filters
    assert config.chunks == chunks
    assert config.blocks == blocks
    assert config.contiguous is False
    assert config.urlpath is urlpath
    if meta is None:
        assert config.meta == 0
    else:
        assert config.meta == meta


@pytest.mark.parametrize(
    "favor, chunks, blocks",
    [
        (ia.Favor.BALANCE, None, None),
        (ia.Favor.SPEED, [50, 50], [20, 20]),
        (ia.Favor.CRATIO, [50, 50], [20, 20]),
    ],
)
def test_global_favor(favor, chunks, blocks):
    ia.set_config_defaults(favor=favor, chunks=chunks, blocks=blocks)
    config = ia.get_config_defaults()
    assert config.favor == favor
    assert config.btune is True
    assert config.chunks == chunks
    assert config.blocks == blocks


@pytest.mark.parametrize(
    "favor",
    [
        ia.Favor.BALANCE,
        ia.Favor.SPEED,
        ia.Favor.CRATIO,
    ],
)
def test_favor_nobtune(favor):
    with pytest.raises(ValueError):
        ia.set_config_defaults(favor=favor, btune=False)

    ia.set_config_defaults(btune=False)
    with pytest.raises(ValueError):
        ia.Config(favor=favor)
    ia.Config(favor=favor, btune=True)


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
        ia.set_config_defaults(clevel=clevel, codec=codec, filters=filters, btune=True)

    ia.set_config_defaults(btune=True)
    with pytest.raises(ValueError):
        ia.Config(clevel=clevel)
    with pytest.raises(ValueError):
        ia.Config(filters=filters)
    with pytest.raises(ValueError):
        ia.Config(codec=codec)


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
        cfg = ia.set_config_defaults(shape=shape, chunks=chunks, blocks=blocks)

        if chunks is not None:
            assert cfg.chunks == chunks
            assert cfg.blocks == blocks
        else:
            # automatic partitioning
            assert cfg.chunks <= shape
            assert cfg.blocks <= cfg.chunks
    except ValueError:
        assert shape == ()

    # One can pass store parameters straight to config() dataclass too
    try:
        cfg = ia.set_config_defaults(
            shape=shape,
            chunks=chunks,
            blocks=blocks,
            contiguous=True,
        )

        if chunks is not None:
            assert cfg.chunks == chunks
            assert cfg.blocks == blocks
        else:
            # automatic partitioning
            assert cfg.chunks <= shape
            assert cfg.blocks <= cfg.chunks
        assert cfg.contiguous is True
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
        with ia.config(
            clevel=clevel, codec=codec, filters=filters, chunks=chunks, blocks=blocks, btune=False
        ) as cfg:
            assert cfg.clevel == clevel
            assert cfg.codec == codec
            assert cfg.btune is False
            assert cfg.filters == filters
            assert cfg.chunks == chunks
            assert cfg.blocks == blocks
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
            assert cfg.btune is False
            assert cfg.filters == filters
            assert cfg.chunks == chunks
            assert cfg.blocks == blocks
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
        with ia.config(shape=shape, chunks=chunks, blocks=blocks) as cfg:
            if chunks is not None:
                assert cfg.chunks == chunks
                assert cfg.blocks == blocks
            else:
                # automatic partitioning
                assert cfg.chunks <= shape
                assert cfg.blocks <= cfg.chunks
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
            if chunks is not None:
                assert cfg.chunks == chunks
                assert cfg.blocks == blocks
            else:
                # automatic partitioning
                assert cfg.chunks <= shape
                assert cfg.blocks <= cfg.chunks
            assert cfg.contiguous is True
    except ValueError:
        assert shape == ()


def test_nested_contexts():
    # Set the default to enable compression
    ia.set_config_defaults(clevel=5, btune=False)
    a = ia.ones((100, 100))
    b = a.data

    assert a.cratio > 1

    # Now play with contexts and params in calls
    # Disable compression in contexts
    with ia.config(clevel=0):
        a = ia.numpy2iarray(b)
        assert a.cratio < 1
        # Enable compression in call
        a = ia.numpy2iarray(b, clevel=1, split_mode=ia.SplitMode.NEVER_SPLIT)
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

    cfg = ia.get_config_defaults()
    ia.set_config_defaults(split_mode=ia.SplitMode.ALWAYS_SPLIT, mode="w")
    a = ia.linspace([10], start=0, stop=1, urlpath=urlpath, contiguous=False)
    ia.set_config_defaults(split_mode=cfg.split_mode)
    cfg2 = ia.Config()

    assert cfg.split_mode == cfg2.split_mode
    assert cfg.contiguous == cfg2.contiguous
    assert cfg.urlpath == cfg2.urlpath
    assert cfg.mode == cfg2.mode
    ia.remove_urlpath(urlpath)

def test_zfp_codec():
    with pytest.raises(ValueError):
        ia.set_config_defaults(btune=False, codec=ia.Codec.ZFP_FIXED_ACCURACY)
    with pytest.raises(ValueError):
        ia.set_config_defaults(btune=False, codec=ia.Codec.ZFP_FIXED_RATE)
    with pytest.raises(ValueError):
        ia.set_config_defaults(btune=False, codec=ia.Codec.ZFP_FIXED_PRECISION)

    with pytest.raises(ValueError):
        ia.set_config_defaults(btune=False, codec=ia.Codec.LZ4, meta=3)


    shape = [100, 100]
    chunks = [30, 30]
    blocks = [2, 2]
    start = 0
    stop = 5
    dtype = np.float32
    contiguous = False
    urlpath = "test_zfp.iarr"
    codec = ia.Codec.ZFP_FIXED_ACCURACY
    meta = -2

    size = int(np.prod(shape))
    ia.remove_urlpath(urlpath)
    print("tira-li al ia.linspace")
    a = ia.linspace(shape, start, stop, dtype=dtype, chunks=chunks, blocks=blocks, contiguous=contiguous,
                    urlpath=urlpath, codec=codec, meta=meta, btune=False)
    b = ia.iarray2numpy(a)
    c = np.linspace(start, stop, size, dtype=dtype).reshape(shape)
    np.testing.assert_allclose(b, c, rtol=1e-2)

    ia.remove_urlpath(urlpath)


@pytest.fixture(scope="function", autouse=True)
def cleanup(request):
    # Make the defaults sane for other tests to come
    return request.addfinalizer(ia.reset_config_defaults)
