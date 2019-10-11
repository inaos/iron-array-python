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
from iarray.high_level import IArray
from collections import namedtuple
from .expression import Parser


cdef class ReadBlockIter:
    cdef ciarray.iarray_iter_read_block_t *_iter
    cdef ciarray.iarray_iter_read_block_value_t _val
    cdef Container _c
    cdef int dtype
    cdef int flag
    cdef object Info

    def __cinit__(self, c, block):
        self._c = c
        cdef ciarray.int64_t block_[ciarray.IARRAY_DIMENSION_MAX]
        if block is None:
            block = c.pshape
        for i in range(len(block)):
            block_[i] = block[i]

        ciarray.iarray_iter_read_block_new(self._c._ctx._ctx, &self._iter, self._c._c, block_, &self._val, False)
        if self._c.dtype == np.float64:
            self.dtype = 0
        else:
            self.dtype = 1
        self.Info = namedtuple('Info', 'index elemindex nblock shape size')

    def __dealloc__(self):
        ciarray.iarray_iter_read_block_free(&self._iter)

    def __iter__(self):
        return self

    def __next__(self):
        if ciarray.iarray_iter_read_block_has_next(self._iter) != 0:
            raise StopIteration

        ciarray.iarray_iter_read_block_next(self._iter, NULL, 0)
        shape = tuple(self._val.block_shape[i] for i in range(self._c.ndim))
        size = np.prod(shape)
        if self.dtype == 0:
            view = <np.float64_t[:size]> self._val.block_pointer
        else:
            view = <np.float32_t[:size]> self._val.block_pointer
        a = np.asarray(view)

        elem_index = tuple(self._val.elem_index[i] for i in range(self._c.ndim))
        index = tuple(self._val.block_index[i] for i in range(self._c.ndim))
        nblock = self._val.nblock
        info = self.Info(index=index, elemindex=elem_index, nblock=nblock, shape=shape, size=size)
        return info, a.reshape(shape)


cdef class WriteBlockIter:
    cdef ciarray.iarray_iter_write_block_t *_iter
    cdef ciarray.iarray_iter_write_block_value_t _val
    cdef Container _c
    cdef int dtype
    cdef int flag
    cdef object Info

    def __cinit__(self, c, block=None):
        self._c = c
        cdef ciarray.int64_t block_[ciarray.IARRAY_DIMENSION_MAX]
        if block is None:
            # The block for iteration has always be provided
            block = c.pshape
        for i in range(len(block)):
            block_[i] = block[i]
        retcode = ciarray.iarray_iter_write_block_new(self._c._ctx._ctx, &self._iter, self._c._c, block_, &self._val,
                                                      False)
        assert(retcode == 0)
        if self._c.dtype == np.float64:
            self.dtype = 0
        else:
            self.dtype = 1
        self.Info = namedtuple('Info', 'index elemindex nblock shape size')

    def __dealloc__(self):
        ciarray.iarray_iter_write_block_free(&self._iter)

    def __iter__(self):
        return self

    def __next__(self):
        if ciarray.iarray_iter_write_block_has_next(self._iter) != 0:
            raise StopIteration

        ciarray.iarray_iter_write_block_next(self._iter, NULL, 0)
        shape = tuple(self._val.block_shape[i] for i in range(self._c.ndim))
        size = np.prod(shape)
        if self.dtype == 0:
            view = <np.float64_t[:size]> self._val.block_pointer
        else:
            view = <np.float32_t[:size]> self._val.block_pointer
        a = np.asarray(view)

        elem_index = tuple(self._val.elem_index[i] for i in range(self._c.ndim))
        index = tuple(self._val.block_index[i] for i in range(self._c.ndim))
        nblock = self._val.nblock

        info = self.Info(index=index, elemindex=elem_index, nblock=nblock, shape=shape, size=size)
        return info, a.reshape(shape)


cdef class IArrayInit:
    def __cinit__(self):
        ciarray.iarray_init()

    def __delloc(self):
        ciarray.iarray_destroy()


cdef class _Config:
    cdef ciarray.iarray_config_t _cfg

    def __init__(self, compression_codec=1, compression_level=5, use_dict=0, filter_flags=1,
                 max_num_threads=1, fp_mantissa_bits=0, blocksize=0, eval_flags="iterblock"):
        self._cfg.compression_codec = compression_codec
        self._cfg.compression_level = compression_level
        self._cfg.use_dict = use_dict
        self._cfg.filter_flags = filter_flags
        if eval_flags == "iterblock":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERBLOCK
        elif eval_flags == "iterblosc":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERBLOSC
        elif eval_flags == "iterchunk":
            self._cfg.eval_flags = ciarray.IARRAY_EXPR_EVAL_ITERCHUNK
        else:
            raise ValueError("eval_flags not recognized:", eval_flags)
        self._cfg.max_num_threads = max_num_threads
        self._cfg.fp_mantissa_bits = fp_mantissa_bits
        self._cfg.blocksize = blocksize

    def to_dict(self):
        return <object> self._cfg


cdef class Context:
    cdef ciarray.iarray_context_t *_ctx

    def __init__(self, cfg):
        cdef ciarray.iarray_config_t cfg_ = cfg.to_dict()
        ciarray.iarray_context_new(&cfg_, &self._ctx)

    def __dealloc__(self):
        ciarray.iarray_context_free(&self._ctx)

    def to_capsule(self):
        return PyCapsule_New(self._ctx, "iarray_context_t*", NULL)

    def __str__(self):
        return "IARRAY CONTEXT OBJECT"


cdef class _DTShape:
    cdef ciarray.iarray_dtshape_t _dtshape

    def __cinit__(self, shape, pshape=None, dtype=np.float64):
        self._dtshape.ndim = len(shape)
        if dtype == np.float64:
            self._dtshape.dtype = ciarray.IARRAY_DATA_TYPE_DOUBLE
        elif dtype == np.float32:
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
        dtype = [np.float64, np.float32]
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
        return "IArray random context object"


cdef class Container:
    cdef ciarray.iarray_container_t *_c
    cdef Context _ctx

    def __init__(self, ctx, c):
        if ctx is None:
            raise ValueError("You must pass a context to the Container constructor")
        if c is None:
            raise ValueError("You must pass a Capsule to the C container struct of the Container constructor")
        self._ctx = ctx
        cdef ciarray.iarray_container_t* c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
            c, "iarray_container_t*")
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
        dtype = [np.float64, np.float32]
        cdef ciarray.iarray_dtshape_t dtshape
        ciarray.iarray_get_dtshape(self._ctx._ctx, self._c, &dtshape)
        return dtype[dtshape.dtype]

    @property
    def cratio(self):
        cdef ciarray.int64_t nbytes, cbytes
        ciarray.iarray_container_info(self._c, &nbytes, &cbytes)
        return <double>nbytes / <double>cbytes

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

    def __init__(self, cfg):
        self._ctx = Context(cfg)
        cdef ciarray.iarray_expression_t* e
        ciarray.iarray_expr_new(self._ctx._ctx, &e)
        self._e = e
        self.expression = None

    def __dealloc__(self):
        ciarray.iarray_expr_free(self._ctx._ctx, &self._e)

    def bind(self, var, c):
        var2 = var.encode("utf-8") if isinstance(var, str) else var
        cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
            c.to_capsule(), "iarray_container_t*")
        ciarray.iarray_expr_bind(self._e, var2, c_)

    def compile(self, expr):
        expr = Parser().parse(expr).simplify({}).toString()
        expr2 = expr.encode("utf-8") if isinstance(expr, str) else expr
        if ciarray.iarray_expr_compile(self._e, expr2) != 0:
            raise ValueError(f"Error in compiling expr: {expr}")
        self.expression = expr2

    def compile_udf(self, bc_udf):
        if ciarray.iarray_expr_compile_udf(self._e, len(bc_udf), bc_udf) != 0:
            raise ValueError(f"Error in compiling udf...")
        self.expression = "user_defined_function"

    def eval(self, shape, pshape=None, dtype=np.float64, filename=None):
        dtshape = _DTShape(shape, pshape, dtype).to_dict()
        cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape
        cdef ciarray.iarray_store_properties_t store
        if filename is not None:
            filename = filename.encode("utf-8") if isinstance(filename, str) else filename
            store.id = filename

        flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST
        cdef ciarray.iarray_container_t *c
        ctx_ = self._ctx._ctx
        if flags == ciarray.IARRAY_CONTAINER_PERSIST:
            ciarray.iarray_container_new(ctx_, &dtshape_, &store, flags, &c)
        else:
            ciarray.iarray_container_new(ctx_, &dtshape_, NULL, flags, &c)

        if ciarray.iarray_eval(self._e, c) != 0:
            raise ValueError(f"Error in evaluating expr: {self.expression}")

        c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
        return IArray(self._ctx, c_c)

#
# Iarray container constructors
#

def partition_advice(ctx, dtshape, low=128*1024, high=1024*1024):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape
    ciarray.iarray_partition_advice(ctx_, &dtshape_, low, high)
    return dict(dtshape_)["pshape"]

def copy(cfg, src, view=False, filename=None):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST
    cdef ciarray.iarray_container_t *c
    cdef ciarray.iarray_container_t *src_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(
        src.to_capsule(), "iarray_container_t*")
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_copy(ctx_, src_, view, &store, flags, &c)
    else:
        ciarray.iarray_copy(ctx_, src_, view, NULL, flags, &c)
    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def empty(cfg, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")
    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def arange(cfg, slice, shape=None, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    start, stop, step = slice.start, slice.stop, slice.step
    if shape is None:
        shape = [ceil((stop - start)/step)]
    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def linspace(cfg, nelem, start, stop, shape=None, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    if shape is None:
        shape = [nelem]
    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def zeros(cfg, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")
    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def ones(cfg, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")
    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def full(cfg, fill_value, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")
    dtshape = _DTShape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        if dtype == np.float64:
            ciarray.iarray_fill_double(ctx_, &dtshape_, fill_value, &store, flags, &c)
        else:
            ciarray.iarray_fill_float(ctx_, &dtshape_, fill_value, &store, flags, &c)
    else:
        if dtype == np.float64:
            ciarray.iarray_fill_double(ctx_, &dtshape_, fill_value, NULL, flags, &c)
        else:
            ciarray.iarray_fill_float(ctx_, &dtshape_, fill_value, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def _get_slice(ctx, data, start, stop, pshape=None, filename=None, view=True):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *data_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
        data.to_capsule(), "iarray_container_t*")

    shape = [sp%s - st%s for sp, st, s in zip(stop, start, data.shape)]
    if pshape is None:
        pshape = partition_advice(ctx, _DTShape(shape, data.pshape, data.dtype).to_dict())

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

    b =  IArray(ctx, c_c)
    return b


def numpy2iarray(cfg, a, pshape=None, filename=None):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    dtype = None
    if a.dtype == np.float64:
        dtype = np.float64
    elif a.dtype == np.float32:
        dtype = np.float32
    else:
        raise NotImplementedError("Only float32 and float64 types are supported for now")

    dtshape = _DTShape(a.shape, pshape, dtype).to_dict()
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


def iarray2numpy(cfg, c):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
        c.to_capsule(), "iarray_container_t*")

    cdef ciarray.iarray_dtshape_t dtshape
    ciarray.iarray_get_dtshape(ctx_, c_, &dtshape)
    shape = []
    for i in range(dtshape.ndim):
        shape.append(dtshape.shape[i])
    size = np.prod(shape, dtype=np.int64)

    npdtype = np.float64 if dtshape.dtype == ciarray.IARRAY_DATA_TYPE_DOUBLE else np.float32
    if ciarray.iarray_is_empty(c_):
        # Return an empty array.  Another possibility would be to raise an exception here?  Let's wait for a use case...
        return np.empty(size, dtype=npdtype).reshape(shape)

    a = np.zeros(size, dtype=npdtype).reshape(shape)
    ciarray.iarray_to_buffer(ctx_, c_, np.PyArray_DATA(a), size * sizeof(npdtype))
    return a


def from_file(cfg, filename, load_in_mem=False):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    cdef ciarray.iarray_store_properties_t store
    filename = filename.encode("utf-8") if isinstance(filename, str) else filename
    store.id = filename

    cdef ciarray.iarray_container_t *c
    ciarray.iarray_from_file(ctx_, &store, &c, load_in_mem)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)

#
# Expression functions
#

def expr_bind(e, var, c):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(
        e, "iarray_expression_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
        c.to_capsule(), "iarray_container_t*")
    ciarray.iarray_expr_bind(e_, var, c_)


def expr_compile(e, expr):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(
        e, "iarray_expression_t*")
    ciarray.iarray_expr_compile(e_, expr)


def expr_eval(e, c):
    cdef ciarray.iarray_expression_t* e_= <ciarray.iarray_expression_t*> PyCapsule_GetPointer(
        e, "iarray_expression_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
        c.to_capsule(), "iarray_container_t*")
    ciarray.iarray_eval(e_, c_)

#
# Random functions
#

def random_rand(cfg, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def random_randn(cfg, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def random_beta(cfg, alpha, beta, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_ALPHA, alpha)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_ALPHA, alpha)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def random_lognormal(cfg, mu, sigma, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma)

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def random_exponential(cfg, beta, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def random_uniform(cfg, a, b, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_A, a)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_B, b)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_A, a)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_B, b)

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def random_normal(cfg, mu, sigma, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma)

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_store_properties_t store
    if filename is not None:
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        store.id = filename

    flags = 0 if filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if flags == ciarray.IARRAY_CONTAINER_PERSIST:
        ciarray.iarray_random_normal(ctx_, &dtshape_, r_ctx_, &store, flags, &c)
    else:
        ciarray.iarray_random_normal(ctx_, &dtshape_, r_ctx_, NULL, flags, &c)

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return IArray(ctx, c_c)


def random_bernoulli(cfg, p, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p)

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def random_binomial(cfg, m, p, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_M, m)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_M, m)

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def random_poisson(cfg, l, shape, pshape=None, dtype=np.float64, filename=None):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_LAMBDA, l)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_LAMBDA, l)

    dtshape = _DTShape(shape, pshape, dtype).to_dict()
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


def random_kstest(cfg, a, b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.bool res;
    ciarray.iarray_random_kstest(ctx_, a_, b_, &res)
    return res


def matmul(cfg, a, b, block_a, block_b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    if a.pshape is None and b.pshape is None:
        if len(b.shape) == 1:
            dtshape = _DTShape(tuple([a.shape[0]]), None, a.dtype).to_dict()
        else:
            dtshape = _DTShape((a.shape[0], b.shape[1]), None, a.dtype).to_dict()
    else:
        if len(b.shape) == 1:
            dtshape = _DTShape(tuple([a.shape[0]]), tuple([block_a[0]]), a.dtype).to_dict()
        else:
            dtshape = _DTShape((a.shape[0], b.shape[1]), (block_a[0], block_b[1]), a.dtype).to_dict()

    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape
    ciarray.iarray_container_new(ctx_, &dtshape_, NULL, 0, &c)

    cdef ciarray.int64_t *block_a_
    cdef ciarray.int64_t *block_b_

    block_a_ = <ciarray.int64_t*> malloc(a.ndim * sizeof(ciarray.int64_t))
    for i in range(a.ndim):
        block_a_[i] = block_a[i]

    block_b_ = <ciarray.int64_t*> malloc(a.ndim * sizeof(ciarray.int64_t))
    for i in range(b.ndim):
        block_b_[i] = block_b[i]


    err = ciarray.iarray_linalg_matmul(ctx_, a_, b_, c, block_a_, block_b_, ciarray.IARRAY_OPERATOR_GENERAL)
    if err != 0:
        raise AttributeError

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
    shape = xa.shape
    cdef np.ndarray[np.npy_float64] y = np.empty(xa.shape, xa.dtype).flatten()
    cdef np.ndarray[np.npy_float64] x = xa.flatten()
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return y.reshape(shape)


# from cython.parallel import prange
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void poly_nogil(double *x, double *y, int n) nogil:
    cdef int i
    # for i in prange(n):
    for i in range(n):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)


def poly_cython_nogil(xa):
    shape = xa.shape
    cdef np.ndarray[np.npy_float64] y = np.empty(xa.shape, xa.dtype).flatten()
    cdef np.ndarray[np.npy_float64] x = xa.flatten()
    poly_nogil(&x[0], &y[0], len(x))
    return y.reshape(shape)

# TODO: End of the benchmarking code
