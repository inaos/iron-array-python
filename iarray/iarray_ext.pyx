cimport ciarray_ext as ciarray
import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer


def init():
    err = ciarray.iarray_init()


def destroy():
    ciarray.iarray_destroy()


def config_new(compression_codec=1, compression_level=5, filter_flags=0, eval_flags=0, max_num_threads=1,
               fp_mantissa_bits=0, blocksize=0):

    cdef ciarray.iarray_config_t cfg
    cfg.compression_codec = compression_codec
    cfg.compression_level = compression_level
    cfg.filter_flags = filter_flags
    cfg.eval_flags = eval_flags
    cfg.max_num_threads = max_num_threads
    cfg.fp_mantissa_bits = fp_mantissa_bits
    cfg.blocksize = blocksize

    return <object> cfg


def context_new(cfg):

    cdef ciarray.iarray_config_t cfg_ = <ciarray.iarray_config_t> cfg

    cdef ciarray.iarray_context_t* ctx
    ciarray.iarray_context_new(&cfg_, &ctx)

    return PyCapsule_New(ctx, "iarray_context_t*", NULL)


def context_free(ctx):
    cdef ciarray.iarray_context_t* ctx_= <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")

    ciarray.iarray_context_free(&ctx_)


def dtshape_new(shape, pshape=None, type="double"):

    cdef ciarray.iarray_dtshape_t dtshape
    dtshape.ndim = len(shape)
    if type == "double":
        dtshape.dtype = ciarray.IARRAY_DATA_TYPE_DOUBLE
    elif type == "float":
        dtshape.dtype = ciarray.IARRAY_DATA_TYPE_FLOAT
    for i in range(len(shape)):
        dtshape.shape[i] = shape[i]
        if pshape != None:
            dtshape.pshape[i] = pshape[i]
        else:
            dtshape.pshape[i] = shape[i]

    return <object> dtshape


def container_new(ctx, dtshape, filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_container_new(ctx_, &dtshape_, &store, flags, &c)
    else:
        ciarray.iarray_container_new(ctx_, &dtshape_, NULL, flags, &c)

    return PyCapsule_New(c, "iarray_container_t*", NULL)


def arange(ctx, dtshape, start, stop, step):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_container_t *c
    ciarray.iarray_arange(ctx_, &dtshape_, start, stop, step, NULL, 0, &c)

    return PyCapsule_New(c, "iarray_container_t*", NULL)


def iarray2numpy(ctx, c):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t*")

    cdef ciarray.iarray_dtshape_t *dtshape
    ciarray.iarray_container_dtshape(ctx_, c_, &dtshape)

    shape = []
    for i in range(dtshape.ndim):
        shape.append(dtshape.shape[i])
    size = np.prod(shape, dtype=np.int64)

    a = np.zeros(size, dtype=np.float64).reshape(shape)
    ciarray.iarray_to_buffer(ctx_, c_, np.PyArray_DATA(a), size)

    return a


def numpy2iarray(ctx, a, pshape=None, filename=None):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")

    cdef ciarray.iarray_store_properties_t store
    if filename != None:
        store.id = filename

    flags = 0 if filename == None else ciarray.IARRAY_CONTAINER_PERSIST

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

    buffer_size = a.size
    if a.dtype == np.float64:
       buffer_size *= 8
    elif a.dtype == np.float32:
        buffer_size *= 4

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_from_buffer(ctx_, &dtshape, np.PyArray_DATA(a), buffer_size, &store, flags, &c)
    else:
        ciarray.iarray_from_buffer(ctx_, &dtshape, np.PyArray_DATA(a), buffer_size, NULL, flags, &c)

    return PyCapsule_New(c, "iarray_container_t*", NULL)


def from_file(ctx, filename):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")

    cdef ciarray.iarray_store_properties_t store
    store.id = filename

    cdef ciarray.iarray_container_t *c
    ciarray.iarray_from_file(ctx_, &store, &c)

    return PyCapsule_New(c, "iarray_container_t*", NULL)


def container_free(ctx, c):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t*")

    ciarray.iarray_container_free(ctx_, &c_)


# Expression functions

def expr_new(ctx):
    cdef ciarray.iarray_context_t* ctx_= <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")

    cdef ciarray.iarray_expression_t *expr
    ciarray.iarray_expr_new(ctx_, &expr)

    return PyCapsule_New(expr, "iarray_expression_t*", NULL)

def expr_free(ctx, expr):
    cdef ciarray.iarray_context_t* ctx_= <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
    cdef ciarray.iarray_expression_t* expr_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(expr, "iarray_expression_t*")

    ciarray.iarray_expr_free(ctx_, &expr_)


def expr_bind(expr, var, c):
    cdef ciarray.iarray_expression_t* expr_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(expr, "iarray_expression_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t*")
    ciarray.iarray_expr_bind(expr_, var, c_)

def expr_bind_scalar_float(e, var, c):
    pass

def expr_bind_scalar_double(e, var, c):
    pass

def expr_compile(expr, expression):
    cdef ciarray.iarray_expression_t* expr_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(expr, "iarray_expression_t*")
    ciarray.iarray_expr_compile(expr_, expression)

def expr_eval(expr, c):
    cdef ciarray.iarray_expression_t* expr_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(expr, "iarray_expression_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t*")
    ciarray.iarray_eval(expr_, c_)

