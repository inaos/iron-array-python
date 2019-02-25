import iarray as ia

ia.iarray_init()

cfg = ia.iarray_config_new()

ctx = ia.iarray_context_new(cfg)

dtshape = ia.iarray_dtshape_new(shape=(4, 4), pshape=(2, 2))

a = ia.iarray_arange(ctx, dtshape, 0, 16, 1)

b = ia.iarray_to_buffer(ctx, a)

print(b)

ia.iarray_destroy()
