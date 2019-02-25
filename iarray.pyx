cimport ciarray
import cython
import numpy as np
cimport numpy as np

def iarray_init():
    err = ciarray.iarray_init()
    if err != 0:
        return None

def iarray_destroy():
    ciarray.iarray_destroy()

def iarray_config_new(codec=1, level=5, filter=0, eval=0, threads=1, mantissa=0, blocksize=0):
    cdef ciarray.iarray_config_t cfg
    cfg.compression_codec = codec
    cfg.compression_level = level
    cfg.filter_flags = filter
    cfg.eval_flags = eval
    cfg.max_num_threads = threads
    cfg.fp_mantissa_bits = mantissa
    cfg.blocksize = blocksize
    return <object>cfg

def iarray_context_new(cfg):
    cdef ciarray.iarray_config_t cfg_ = <ciarray.iarray_config_t> cfg
    cdef ciarray.iarray_context_t* ctx
    err =  ciarray.iarray_context_new(&cfg_, &ctx)
    if err != 0:
        return None
    return <object>ctx

def iarray_context_free(ctx):
    cdef ciarray.iarray_context_t* ctx_= <ciarray.iarray_context_t*> ctx
    ciarray.iarray_context_free(&ctx_)

def iarray_dtshape_new(shape, pshape, type="double"):
    cdef ciarray.iarray_dtshape_t dtshape
    dtshape.ndim = len(shape)
    if type == "double":
        dtshape.dtype = ciarray.IARRAY_DATA_TYPE_DOUBLE
    for i in range(len(shape)):
        dtshape.shape[i] = shape[i]
        dtshape.pshape[i] = pshape[i]
    return <object>dtshape


def iarray_arange(ctx, dtshape, start, stop, step):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> ctx
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape
    cdef ciarray.iarray_container_t *c
    err = ciarray.iarray_arange(ctx_, &dtshape_, start, stop, step, NULL, 0, &c)
    if err != 0:
        return None
    return <object>c

def iarray_to_buffer(ctx, c):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> ctx
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> c

    a = np.arange(16, dtype=np.float64)
    size = 16*8
    ciarray.iarray_to_buffer(ctx_, c_, np.PyArray_DATA(a), size)

    return a

