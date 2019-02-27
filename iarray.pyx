cimport ciarray
import cython
import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer

def init():
    err = ciarray.iarray_init()
    if err != 0:
        return None

def destroy():
    ciarray.iarray_destroy()

def config_new(codec=1, level=5, filter=0, eval=0, threads=1, mantissa=0, blocksize=0):
    cdef ciarray.iarray_config_t cfg
    cfg.compression_codec = codec
    cfg.compression_level = level
    cfg.filter_flags = filter
    cfg.eval_flags = eval
    cfg.max_num_threads = threads
    cfg.fp_mantissa_bits = mantissa
    cfg.blocksize = blocksize
    return <object>cfg

def context_new(cfg):
    cdef ciarray.iarray_config_t cfg_ = <ciarray.iarray_config_t> cfg
    cdef ciarray.iarray_context_t* ctx
    err =  ciarray.iarray_context_new(&cfg_, &ctx)
    if err != 0:
        return None
    return PyCapsule_New(ctx, "iarray_context_t", NULL)

def context_free(ctx):
    cdef ciarray.iarray_context_t* ctx_= <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t")
    ciarray.iarray_context_free(&ctx_)

def dtshape_new(shape, pshape, type="double"):
    cdef ciarray.iarray_dtshape_t dtshape
    dtshape.ndim = len(shape)
    if type == "double":
        dtshape.dtype = ciarray.IARRAY_DATA_TYPE_DOUBLE
    for i in range(len(shape)):
        dtshape.shape[i] = shape[i]
        dtshape.pshape[i] = pshape[i]
    return <object>dtshape


def arange(ctx, dtshape, start, stop, step):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t")
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_container_t *c
    err = ciarray.iarray_arange(ctx_, &dtshape_, start, stop, step, NULL, 0, &c)

    if err != 0:
        return None

    return PyCapsule_New(c, "iarray_container_t", NULL)

def iarray2numpy(ctx, c):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t")

    cdef ciarray.iarray_dtshape_t *dtshape

    ciarray.iarray_container_dtshape(ctx_, c_, &dtshape)

    shape = []
    for i in range(dtshape.ndim):
        shape.append(dtshape.shape[i])

    size = np.prod(shape, dtype=np.int64)

    a = np.zeros(size, dtype=np.float64).reshape(shape)

    err = ciarray.iarray_to_buffer(ctx_, c_, np.PyArray_DATA(a), size)

    return a

def numpy2iarray(ctx, a, pshape=None, filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t")

    cdef ciarray.iarray_store_properties_t store

    flags = 0 if filename == None else ciarray.IARRAY_CONTAINER_PERSIST

    if filename != None:
        store.id = filename

    cdef ciarray.iarray_dtshape_t dtshape
    dtshape.ndim = a.ndim
    if a.dtype == np.float64:
        dtshape.dtype = ciarray.IARRAY_DATA_TYPE_DOUBLE
    elif a.dtype == np.float32:
        dtshape.dtype = ciarray.IARRAY_DATA_TYPE_FLOAT
    for i in range(len(a.shape)):
        dtshape.shape[i] = a.shape[i]
        if pshape != None:
            dtshape.pshape[i] = pshape[i]
        else:
            dtshape.pshape[i] = a.shape[i]

    cdef ciarray.iarray_container_t *c

    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_from_buffer(ctx_, &dtshape, np.PyArray_DATA(a), a.size * 8, &store, flags, &c)
    else:
        ciarray.iarray_from_buffer(ctx_, &dtshape, np.PyArray_DATA(a), a.size * 8, NULL, flags, &c)

    return PyCapsule_New(c, "iarray_container_t", NULL)

def from_file(ctx, filename):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t")
    cdef ciarray.iarray_container_t *c
    cdef ciarray.iarray_store_properties_t store
    store.id = filename
    ciarray.iarray_from_file(ctx_, &store, &c)
    return PyCapsule_New(c, "iarray_container_t", NULL)

def container_free(ctx, c):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t")
    ciarray.iarray_container_free(ctx_, &c_)
