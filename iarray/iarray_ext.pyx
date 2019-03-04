cimport ciarray_ext as ciarray
import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from math import ceil


cdef class Config:
    cdef ciarray.iarray_config_t _cfg

    def __cinit__(self, compression_codec=1, compression_level=5, filter_flags=0, eval_flags="iterblock",
                max_num_threads=1, fp_mantissa_bits=0, blocksize=0):
        self._cfg.compression_codec = compression_codec
        self._cfg.compression_level = compression_level
        self._cfg.filter_flags = filter_flags
        if eval_flags == "iterblock":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERBLOCK
        elif eval_flags == "iterchunk":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERCHUNK
        elif eval_flags == "chunk":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_CHUNK
        else:
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_BLOCK
        self._cfg.max_num_threads = max_num_threads
        self._cfg.fp_mantissa_bits = fp_mantissa_bits
        self._cfg.blocksize = blocksize

    def _get(self):
        return <object> self._cfg

    @property
    def compression_codec(self):
            codec = ["BloscLZ", "LZ4", "LZ4HC", "Snappy", "Zlib", "Zstd", "Lizard"]
            return codec[self._cfg.compression_codec]

    @property
    def compression_level(self):
        return self._cfg.compression_level

    @property
    def filter_flags(self):
        flags = {0: "No filters", 1: "Shuffle", 2: "Bit Shuffle", 4: "Delta", 8: "Trunc. Precision"}
        return flags[self._cfg.filter_flags]

    @property
    def eval_flags(self):
        flags = {1: "Block", 2: "Chunk", 4: "Block (iter)", 8: "Chunk (iter)"}
        return flags[self._cfg.eval_flags]

    @property
    def max_num_threads(self):
        return self._cfg.max_num_threads

    @property
    def fp_mantissa_bits(self):
        return self._cfg.fp_mantissa_bits

    @property
    def blocksize(self):
        return self._cfg.blocksize

    def __str__(self):
        res = f"IARRAY CONFIG OBJECT\n"
        compression_codec = f"    Compression codec: {self.compression_codec}\n"
        compression_level = f"    Compression level: {self.compression_level}\n"
        filter_flags = f"    Filter flags: {self.filter_flags}\n"
        eval_flags = f"    Eval flags: {self.eval_flags}\n"
        max_num_threads = f"    Max. num. threads: {self.max_num_threads}\n"
        fp_mantissa_bits = f"    Fp mantissa bits: {self.fp_mantissa_bits}\n"
        blocksize = f"    Blocksize: {self.blocksize}"
        return res + compression_codec + compression_level + filter_flags + eval_flags +\
               max_num_threads + fp_mantissa_bits + blocksize


cdef class Dtshape:
    cdef ciarray.iarray_dtshape_t _dtshape

    def __cinit__(self, shape, pshape=None, dtype="double"):
        self._dtshape.ndim = len(shape)
        if dtype == "double":
            self._dtshape.dtype = ciarray.IARRAY_DATA_TYPE_DOUBLE
        elif dtype == "float":
            self._dtshape.dtype = ciarray.IARRAY_DATA_TYPE_FLOAT
        for i in range(len(shape)):
            self._dtshape.shape[i] = shape[i]
            if pshape is not None:
               self._dtshape.pshape[i] = pshape[i]
            else:
                self._dtshape.pshape[i] = shape[i]

    cdef _get(self):
        return <object> self._dtshape

    @property
    def ndim(self):
        return self._dtshape.ndim

    @property
    def dtype(self):
        dtype = ["double", "float"]
        return dtype[self._dtshape.dtype]

    @property
    def shape(self):
        shape = []
        for i in range(self.ndim):
            shape.append(self._dtshape.shape[i])
        return shape

    @property
    def pshape(self):
        pshape = []
        for i in range(self.ndim):
            pshape.append(self._dtshape.pshape[i])
        return pshape

    def __str__(self):
        res = f"IARRAY DTSHAPE OBJECT\n"
        ndim = f"    Dimensions: {self.ndim}\n"
        shape = f"    Shape: {self.shape}\n"
        pshape = f"    Pshape: {self.pshape}\n"
        dtype = f"    Datatype: {self.dtype}"

        return res + ndim + shape + pshape + dtype


cdef class Context:
    cdef ciarray.iarray_context_t *_ctx

    def __cinit__(self, cfg):
        cdef ciarray.iarray_config_t cfg__ = cfg._get()
        cdef ciarray.iarray_config_t cfg_ = cfg__
        ciarray.iarray_context_new(&cfg_, &self._ctx)

    def __dealloc__(self):
        ciarray.iarray_context_free(&self._ctx)

    def _get(self):
        PyCapsule_New(self._ctx, "iarray_context_t*", NULL)

    def __str__(self):
        return "IARRAY CONTEXT OBJECT"


cdef class Container:
    cdef ciarray.iarray_container_t *_c
    cdef ciarray.iarray_context_t *_ctx

    def __cinit__(self, ctx, c):
        ctx__ = ctx._get()
        cdef ciarray.iarray_context_t* ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
        self._ctx = ctx_

        cdef ciarray.iarray_container_t* c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t*")
        self._c = c_

    def __dealloc__(self):
        ciarray.iarray_container_free(self._ctx, &self._c)

def init():
    err = ciarray.iarray_init()

def destroy():
    ciarray.iarray_destroy()


def config_new(compression_codec=1, compression_level=5, filter_flags=0, eval_flags="iterblock", max_num_threads=1,
               fp_mantissa_bits=0, blocksize=0):

    cdef ciarray.iarray_config_t cfg
    cfg.compression_codec = compression_codec
    cfg.compression_level = compression_level
    cfg.filter_flags = filter_flags
    if eval_flags == "iterblock":
        cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERBLOCK
    elif eval_flags == "iterchunk":
        cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERCHUNK
    elif eval_flags == "chunk":
        cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_CHUNK
    else:
        cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_BLOCK
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


def _dtshape_new(shape, pshape=None, dtype="double"):

    cdef ciarray.iarray_dtshape_t dtshape
    dtshape.ndim = len(shape)
    if dtype == "double":
        dtshape.dtype = ciarray.IARRAY_DATA_TYPE_DOUBLE
    elif dtype == "float":
        dtshape.dtype = ciarray.IARRAY_DATA_TYPE_FLOAT
    for i in range(len(shape)):
        dtshape.shape[i] = shape[i]
        if pshape is not None:
            dtshape.pshape[i] = pshape[i]
        else:
            dtshape.pshape[i] = shape[i]

    return <object> dtshape


def container_new(ctx, shape, pshape=None, dtype="double", filename=None):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
    dtshape = _dtshape_new(shape, pshape, dtype)
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

def container_free(ctx, c):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t*")

    ciarray.iarray_container_free(ctx_, &c_)

# Iarray container creators

def arange(ctx, *args, shape=None, pshape=None, dtype="double", filename=None):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")

    s = slice(*args)
    start = 0 if s.start is None else s.start
    stop = s.stop
    step = 1 if s.step is None else s.step

    if shape is None:
        shape = [ceil((stop - start)/step)]

    dtshape = _dtshape_new(shape, pshape, dtype)
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_arange(ctx_, &dtshape_, start, stop, step, &store, flags, &c)
    else:
        ciarray.iarray_arange(ctx_, &dtshape_, start, stop, step, NULL, flags, &c)

    return PyCapsule_New(c, "iarray_container_t*", NULL)


def linspace(ctx, nelem, start, stop, shape=None, pshape=None, dtype="double", filename=None):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")

    if shape is None:
        shape = [nelem]

    dtshape = _dtshape_new(shape, pshape, dtype)
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_linspace(ctx_, &dtshape_, nelem, start, stop, &store, flags, &c)
    else:
        ciarray.iarray_linspace(ctx_, &dtshape_, nelem, start, stop, NULL, flags, &c)

    return PyCapsule_New(c, "iarray_container_t*", NULL)

'''
def get_slice(ctx, data, start, stop, pshape=None, filename=None, view=True):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
    cdef ciarray.iarray_container_t *data_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(data, "iarray_container_t*")

    shape = [sp - st for sp, st in zip(stop, start)]

    if pshape is None:
        pshape = shape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST
    cdef ciarray.int64_t *start_ = <ciarray.int64_t*> start
    cdef ciarray.int64_t *stop_ = <ciarray.int64_t*> stop
    cdef ciarray.uint64_t *pshape_ = <ciarray.uint64_t*> pshape
    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_get_slice(ctx_, data_, start_, stop_, pshape_, &store, flags, view, &c)
    else:
        ciarray.iarray_get_slice(ctx_, data_, start_, stop_, pshape_, NULL, flags, view, &c)

    return PyCapsule_New(c, "iarray_container_t*", NULL)
'''

def numpy2iarray(ctx, a, pshape=None, filename=None):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")

    dtype = None
    if a.dtype == np.float64:
        dtype = "double"
    elif a.dtype == np.float32:
        dtype = "float"
    else:
        print("ERROR")

    cdef ciarray.iarray_dtshape_t dtshape = _dtshape_new(a.shape, pshape, dtype)
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    buffer_size = a.size * np.dtype(a.dtype).itemsize

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_from_buffer(ctx_, &dtshape, np.PyArray_DATA(a), buffer_size, &store, flags, &c)
    else:
        ciarray.iarray_from_buffer(ctx_, &dtshape, np.PyArray_DATA(a), buffer_size, NULL, flags, &c)

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

def from_file(ctx, filename):

    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")

    cdef ciarray.iarray_store_properties_t store
    store.id = filename

    cdef ciarray.iarray_container_t *c
    ciarray.iarray_from_file(ctx_, &store, &c)

    return PyCapsule_New(c, "iarray_container_t*", NULL)

# Expression functions

def expr_new(ctx):
    cdef ciarray.iarray_context_t* ctx_= <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")

    cdef ciarray.iarray_expression_t *e
    ciarray.iarray_expr_new(ctx_, &e)

    return PyCapsule_New(e, "iarray_expression_t*", NULL)

def expr_free(ctx, e):
    cdef ciarray.iarray_context_t* ctx_= <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx, "iarray_context_t*")
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(e, "iarray_expression_t*")

    ciarray.iarray_expr_free(ctx_, &e_)


def expr_bind(e, var, c):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(e, "iarray_expression_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t*")
    ciarray.iarray_expr_bind(e_, var, c_)

def expr_bind_scalar_double(e, var, d):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(e, "iarray_expression_t*")
    ciarray.iarray_expr_bind_scalar_double(e_, var, d)

def expr_compile(e, expr):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(e, "iarray_expression_t*")
    ciarray.iarray_expr_compile(e_, expr)

def expr_eval(e, c):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(e, "iarray_expression_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t*")
    ciarray.iarray_eval(e_, c_)
