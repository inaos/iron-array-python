# Hey Cython, this is Python 3!
# cython: language_level=3

###########################################################################################
# Copyright INAOS GmbH, Thalwil, 2018.
# Copyright Francesc Alted, 2018.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of INAOS GmbH
# and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
# Information and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

from . cimport ciarray_ext as ciarray
import numpy as np
cimport numpy as np
import cython
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from math import ceil
from libc.stdlib cimport malloc, free
from iarray.container import IArray


cdef class ReadBlockIter:
    cdef ciarray.iarray_iter_read_block_t *_iter
    cdef ciarray.iarray_iter_read_block_value_t _val
    cdef Container _c
    cdef int dtype
    cdef int flag

    def __cinit__(self, c, block):
        self._c = c
        cdef ciarray.int64_t block_[ciarray.IARRAY_DIMENSION_MAX]
        for i in range(len(block)):
            block_[i] = block[i]

        ciarray.iarray_iter_read_block_new(self._c._ctx._ctx, &self._iter, self._c._c, block_, &self._val)
        if self._c.dtype == "double":
            self.dtype = 0
        else:
            self.dtype = 1

    def __dealloc__(self):
        ciarray.iarray_iter_read_block_free(self._iter)


    def __iter__(self):
        return self

    def __next__(self):
        if not ciarray.iarray_iter_read_block_has_next(self._iter):
            raise StopIteration

        ciarray.iarray_iter_read_block_next(self._iter)

        shape = tuple(self._val.block_shape[i] for i in range(self._c.ndim))
        size = np.prod(shape)

        if self.dtype == 0:
            view = <np.float64_t[:size]> self._val.pointer
        else:
            view = <np.float32_t[:size]> self._val.pointer
        a = np.asarray(view)

        index = tuple(self._val.elem_index[i] for i in range(self._c.ndim))

        return index, a.reshape(shape)

cdef class WriteBlockIter:
    cdef ciarray.iarray_iter_write_block_t *_iter
    cdef ciarray.iarray_iter_write_block_value_t _val
    cdef Container _c
    cdef int dtype
    cdef int flag

    def __cinit__(self, c, block=None):
        self._c = c
        cdef ciarray.int64_t block_[ciarray.IARRAY_DIMENSION_MAX]
        if block is None:
            ciarray.iarray_iter_write_block_new(self._c._ctx._ctx,  &self._iter,self._c._c, NULL, &self._val)
        else:
            for i in range(len(block)):
                block_[i] = block[i]
            ciarray.iarray_iter_write_block_new(self._c._ctx._ctx, &self._iter, self._c._c, block_, &self._val)
        if self._c.dtype == "double":
            self.dtype = 0
        else:
            self.dtype = 1

    def __dealloc__(self):
        ciarray.iarray_iter_write_block_free(self._iter)

    def __iter__(self):
        return self

    def __next__(self):
        if not ciarray.iarray_iter_write_block_has_next(self._iter):
            raise StopIteration

        ciarray.iarray_iter_write_block_next(self._iter)

        shape = tuple(self._val.block_shape[i] for i in range(self._c.ndim))
        size = np.prod(shape)

        if self.dtype == 0:
            view = <np.float64_t[:size]> self._val.pointer
        else:
            view = <np.float32_t[:size]> self._val.pointer
        a = np.asarray(view)

        index = tuple(self._val.elem_index[i] for i in range(self._c.ndim))

        return index, a.reshape(shape)

cdef class IarrayInit:
    def __cinit__(self):
        ciarray.iarray_init()

    def __delloc(self):
        ciarray.iarray_destroy()


cdef class Config:
    cdef ciarray.iarray_config_t _cfg

    def __init__(self, compression_codec=1, compression_level=5, use_dict=0, filter_flags=0, eval_flags="iterblock",
                 max_num_threads=1, fp_mantissa_bits=0, blocksize=0):
        self._cfg.compression_codec = compression_codec
        self._cfg.compression_level = compression_level
        self._cfg.use_dict = use_dict
        self._cfg.filter_flags = filter_flags
        if eval_flags == "iterblock":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERBLOCK
        elif eval_flags == "iterchunk":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERCHUNK
        elif eval_flags == "chunk":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_CHUNK
        elif eval_flags == "block":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_BLOCK
        # else:     // Uncomment this when ITERCHUNKPARA would be in IronArray master
        #     self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERCHUNKPARA
        self._cfg.max_num_threads = max_num_threads
        self._cfg.fp_mantissa_bits = fp_mantissa_bits
        self._cfg.blocksize = blocksize

    def to_dict(self):
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
        flags = {1: "Block", 2: "Chunk", 4: "Block (iter)", 8: "Chunk (iter)", 16: "Chunk (iterpara)"}
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


cdef class _Dtshape:
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
                self._dtshape.pshape[i] = 0

    cdef to_dict(self):
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
        return tuple(shape)

    @property
    def pshape(self):
        pshape = []
        for i in range(self.ndim):
            pshape.append(self._dtshape.pshape[i])
        return tuple(pshape)

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
        cdef ciarray.iarray_config_t cfg_ = cfg.to_dict()
        ciarray.iarray_context_new(&cfg_, &self._ctx)

    def __dealloc__(self):
        ciarray.iarray_context_free(&self._ctx)

    def to_capsule(self):
        return PyCapsule_New(self._ctx, "iarray_context_t*", NULL)

    def __str__(self):
        return "IARRAY CONTEXT OBJECT"


cdef class RandomContext:
    cdef ciarray.iarray_random_ctx_t *_r_ctx
    cdef Context _ctx

    def __init__(self, ctx, seed=0, rng="MERSENNE_TWISTER"):
        self._ctx = ctx
        cdef ciarray.iarray_random_ctx_t* r_ctx
        if rng == "MERSENNE_TWISTER":
            ciarray.iarray_random_ctx_new(self._ctx._ctx, seed, ciarray.IARRAY_RANDOM_RNG_MERSENNE_TWISTER, &r_ctx)
        else:
            ciarray.iarray_random_ctx_new(self._ctx._ctx, seed, ciarray.IARRAY_RANDOM_RNG_SOBOL, &r_ctx)
        self._r_ctx = r_ctx

    def __dealloc__(self):
        ciarray.iarray_random_ctx_free(self._ctx._ctx, &self._r_ctx)

    def to_capsule(self):
        return PyCapsule_New(self._r_ctx, "iarray_random_ctx_t*", NULL)

    def __str__(self):
        return "IARRAY RANDOM CONTEXT OBJECT"


cdef class Container:
    cdef ciarray.iarray_container_t *_c
    cdef Context _ctx

    def __init__(self, ctx, c):
        self._ctx = ctx
        cdef ciarray.iarray_container_t* c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, "iarray_container_t*")
        self._c = c_

    def __dealloc__(self):
        ciarray.iarray_container_free(self._ctx._ctx, &self._c)

    def iter_read_block(self, block=None):
        return ReadBlockIter(self, block)

    def iter_write_block(self, block=None):
        return WriteBlockIter(self, block)

    def to_capsule(self):
        return PyCapsule_New(self._c, "iarray_container_t*", NULL)

    @property
    def ndim(self):
        cdef ciarray.iarray_dtshape_t dtshape
        ciarray.iarray_get_dtshape(self._ctx._ctx, self._c, &dtshape)

        return dtshape.ndim

    @property
    def shape(self):
        cdef ciarray.iarray_dtshape_t dtshape
        ciarray.iarray_get_dtshape(self._ctx._ctx, self._c, &dtshape)

        shape = [dtshape.shape[i] for i in range(self.ndim)]
        return tuple(shape)

    @property
    def pshape(self):
        cdef ciarray.iarray_dtshape_t dtshape
        ciarray.iarray_get_dtshape(self._ctx._ctx, self._c, &dtshape)

        pshape = [dtshape.pshape[i] for i in range(self.ndim)]
        return tuple(pshape)

    @property
    def dtype(self):
        dtype = ["double", "float"]
        cdef ciarray.iarray_dtshape_t dtshape
        ciarray.iarray_get_dtshape(self._ctx._ctx, self._c, &dtshape)

        return dtype[dtshape.dtype]

    def __str__(self):
        res = f"IARRAY CONTAINER OBJECT\n"
        ndim = f"    Dimensions: {self.ndim}\n"
        shape = f"    Shape: {self.shape}\n"
        pshape = f"    Pshape: {self.pshape}\n"
        dtype = f"    Datatype: {self.dtype}"

        return res + ndim + shape + pshape + dtype

    def __getitem__(self, item):
        if self.ndim == 1:
            item = [item]

        start = [s.start if s.start is not None else 0 for s in item]
        stop = [s.stop if s.stop is not None else sh for s, sh in zip(item, self.shape)]

        return _get_slice(self._ctx, self, start, stop)


cdef class Expression:
    cdef object expression
    cdef ciarray.iarray_expression_t *_e
    cdef Context _ctx

    def __init__(self, ctx):
        self._ctx = ctx
        cdef ciarray.iarray_expression_t* e
        ciarray.iarray_expr_new(self._ctx._ctx, &e)
        self._e = e
        self.expression = None

    def __dealloc__(self):
        ciarray.iarray_expr_free(self._ctx._ctx, &self._e)

    def bind(self, var, c):
        var2 = var.encode("utf-8") if isinstance(var, str) else var
        cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c.to_capsule(), "iarray_container_t*")
        ciarray.iarray_expr_bind(self._e, var2, c_)

    def compile(self, expr):
        expr2 = expr.encode("utf-8") if isinstance(expr, str) else expr
        if ciarray.iarray_expr_compile(self._e, expr2) != 0:
            raise ValueError(f"Error in compiling expr: {expr}")
        self.expression = expr2

    def eval(self, shape, pshape=None, dtype="double", filename=None):

        dtshape = _Dtshape(shape, pshape, dtype).to_dict()
        cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

        cdef ciarray.iarray_container_t *c
        ciarray.iarray_container_new(self._ctx._ctx, &dtshape_, NULL, 0, &c)
        if ciarray.iarray_eval(self._e, c) != 0:
            raise ValueError(f"Error in evaluating expr: {self.expression}")

        c_c = PyCapsule_New(c, "iarray_container_t*", NULL)

        return IArray(self._ctx, c_c)

#
# Iarray container creators
#

def empty(ctx, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_container_new(ctx_, &dtshape_, &store, flags, &c)
    else:
        ciarray.iarray_container_new(ctx_, &dtshape_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)

    return IArray(ctx, c_c)


def arange(ctx, *args, shape=None, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")

    s = slice(*args)
    start = 0 if s.start is None else s.start
    stop = s.stop
    step = 1 if s.step is None else s.step

    if shape is None:
        shape = [ceil((stop - start)/step)]

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_arange(ctx_, &dtshape_, start, stop, step, &store, flags, &c)
    else:
        ciarray.iarray_arange(ctx_, &dtshape_, start, stop, step, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)

    return IArray(ctx, c_c)


def linspace(ctx, nelem, start, stop, shape=None, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")

    if shape is None:
        shape = [nelem]

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_linspace(ctx_, &dtshape_, nelem, start, stop, &store, flags, &c)
    else:
        ciarray.iarray_linspace(ctx_, &dtshape_, nelem, start, stop, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def zeros(ctx, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_zeros(ctx_, &dtshape_, &store, flags, &c)
    else:
        ciarray.iarray_zeros(ctx_, &dtshape_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def ones(ctx, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_ones(ctx_, &dtshape_, &store, flags, &c)
    else:
        ciarray.iarray_ones(ctx_, &dtshape_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def full(ctx, fill_value, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        if dtype == "double":
            ciarray.iarray_fill_double(ctx_, &dtshape_, fill_value, &store, flags, &c)
        else:
            ciarray.iarray_fill_float(ctx_, &dtshape_, fill_value, &store, flags, &c)
    else:
        if dtype == "double":
            ciarray.iarray_fill_double(ctx_, &dtshape_, fill_value, NULL, flags, &c)
        else:
            ciarray.iarray_fill_float(ctx_, &dtshape_, fill_value, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def _get_slice(ctx, data, start, stop, pshape=None, filename=None, view=True):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *data_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(data.to_capsule(), "iarray_container_t*")

    shape = [sp%s - st%s for sp, st, s in zip(stop, start, data.shape)]

    if pshape is None:
        pshape = shape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST
    cdef ciarray.int64_t start_[ciarray.IARRAY_DIMENSION_MAX]
    cdef ciarray.int64_t stop_[ciarray.IARRAY_DIMENSION_MAX]
    cdef ciarray.int64_t pshape_[ciarray.IARRAY_DIMENSION_MAX]

    for i in range(len(start)):
        start_[i] = start[i]
        stop_[i] = stop[i]
        pshape_[i] = pshape[i]

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_get_slice(ctx_, data_, start_, stop_, pshape_, &store, flags, view, &c)
    else:
        ciarray.iarray_get_slice(ctx_, data_, start_, stop_, pshape_, NULL, flags, view, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def numpy2iarray(ctx, a, pshape=None, filename=None):
    """

    :rtype: object
    """
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")

    dtype = None
    if a.dtype == np.float64:
        dtype = "double"
    elif a.dtype == np.float32:
        dtype = "float"
    else:
        print("ERROR")

    dtshape = _Dtshape(a.shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    buffer_size = a.size * np.dtype(a.dtype).itemsize

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_from_buffer(ctx_, &dtshape_, np.PyArray_DATA(a), buffer_size, &store, flags, &c)
    else:
        ciarray.iarray_from_buffer(ctx_, &dtshape_, np.PyArray_DATA(a), buffer_size, NULL, flags, &c)

    c_c =  PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def iarray2numpy(ctx, c):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c.to_capsule(), "iarray_container_t*")

    cdef ciarray.iarray_dtshape_t dtshape
    ciarray.iarray_get_dtshape(ctx_, c_, &dtshape)

    shape = []
    for i in range(dtshape.ndim):
        shape.append(dtshape.shape[i])
    size = np.prod(shape, dtype=np.int64)

    npdtype = np.float64 if dtshape.dtype == ciarray.IARRAY_DATA_TYPE_DOUBLE else np.float32

    a = np.zeros(size, dtype=npdtype).reshape(shape)
    ciarray.iarray_to_buffer(ctx_, c_, np.PyArray_DATA(a), size*8)

    return a


def from_file(ctx, filename):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")

    cdef ciarray.iarray_store_properties_t store
    filename = filename.encode("utf-8") if isinstance(filename, str) else filename
    store.id = filename

    cdef ciarray.iarray_container_t *c
    ciarray.iarray_from_file(ctx_, &store, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)

#
# Expression functions
#

def expr_bind(e, var, c):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(e, "iarray_expression_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c.to_capsule(), "iarray_container_t*")
    ciarray.iarray_expr_bind(e_, var, c_)

def expr_compile(e, expr):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(e, "iarray_expression_t*")
    ciarray.iarray_expr_compile(e_, expr)

def expr_eval(e, c):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(e, "iarray_expression_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c.to_capsule(), "iarray_container_t*")
    ciarray.iarray_eval(e_, c_)

#
# Random functions
#

def random_rand(ctx, r_ctx, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_rand(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_rand(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def random_randn(ctx, r_ctx, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_randn(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_randn(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def random_beta(ctx, r_ctx, alpha, beta, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == "double":
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_ALPHA, alpha)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_ALPHA, alpha)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_beta(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_beta(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def random_lognormal(ctx, r_ctx, mu, sigma, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == "double":
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma)

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_lognormal(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_lognormal(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def random_exponential(ctx, r_ctx, beta, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == "double":
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_exponential(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_exponential(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def random_uniform(ctx, r_ctx, a, b, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == "double":
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_A, a)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_B, b)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_A, a)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_B, b)

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_uniform(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_uniform(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def random_bernoulli(ctx, r_ctx, p, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == "double":
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p)

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_bernoulli(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_bernoulli(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)

def random_binomial(ctx, r_ctx, m, p, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == "double":
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_M, m)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_M, m)

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_binomial(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_binomial(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)

def random_poisson(ctx, r_ctx, l, shape, pshape=None, dtype="double", filename=None):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == "double":
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_LAMBDA, l)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_LAMBDA, l)

    dtshape = _Dtshape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_poisson(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_poisson(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)

def random_kstest(ctx, a, b):
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.bool res;
    ciarray.iarray_random_kstest(ctx_, a_, b_, &res)
    return res


def matmul(ctx, a, b, block_a, block_b):
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c
    dtshape = _Dtshape((a.shape[1], b.shape[0]), (block_a[1], block_b[0]), a.dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape
    ciarray.iarray_container_new(ctx_, &dtshape_, NULL, 0, &c)

    cdef ciarray.int64_t *block_a_ = <ciarray.int64_t*> malloc(a.ndim * sizeof(ciarray.int64_t))
    cdef ciarray.int64_t *block_b_ = <ciarray.int64_t*> malloc(b.ndim * sizeof(ciarray.int64_t))
    for i in range(a.ndim):
        block_a_[i] = block_a[i]
    for i in range(b.ndim):
        block_b_[i] = block_b[i]
    ciarray.iarray_linalg_matmul(ctx_, a_, b_, c, block_a_, block_b_,ciarray.IARRAY_OPERATOR_GENERAL)
    free(block_a_)
    free(block_b_)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)
#
# TODO: the next functions are just for benchmarking purposes and should be moved to its own extension
#

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def poly_cython(xa):
    cdef np.ndarray[np.npy_float64] y = np.empty(xa.shape, xa.dtype)
    cdef np.ndarray[np.npy_float64] x = xa
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5) * (x[i] + 1.5) * (x[i] + 4.6)
    return y


# from cython.parallel import prange
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void poly_nogil(double *x, double *y, int n) nogil:
    cdef int i
    # for i in prange(n):
    for i in range(n):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5) * (x[i] + 1.5) * (x[i] + 4.6)


def poly_cython_nogil(xa):
    cdef np.ndarray[np.npy_float64] y = np.empty(xa.shape, xa.dtype)
    cdef np.ndarray[np.npy_float64] x = xa
    poly_nogil(&x[0], &y[0], len(x))
    return y

# TODO: End of the benchmarking code
