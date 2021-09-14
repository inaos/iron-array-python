import iarray as ia

shape = ()
chunks = ()
blocks = ()

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
        enforce_frame=True,
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
    ia.set_config(clevel=5)
    a = ia.ones((100, 100))


ia.set_config(clevel=5)

a = ia.ones((100, 100))
