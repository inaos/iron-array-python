import iarray as ia

shape = ()
chunkshape = ()
blockshape = ()

try:
    storage = ia.Store(chunkshape, blockshape)
    dtshape = ia.DTShape(shape)
    with ia.config(dtshape=dtshape, storage=storage) as cfg:
        storage2 = cfg.store
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
        storage2 = cfg.store
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


ia.set_config(clevel=5)

a = ia.ones(ia.DTShape((100, 100)))
