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
import iarray as ia
from collections import namedtuple


class IArrayError(Exception):
    pass


def iarray_check(error):
    if error != 0:
        raise IArrayError(str(ciarray.iarray_err_strerror(error)))


IARRAY_ERR_EVAL_ENGINE_FAILED = ciarray.IARRAY_ERR_EVAL_ENGINE_FAILED
IARRAY_ERR_EVAL_ENGINE_NOT_COMPILED = ciarray.IARRAY_ERR_EVAL_ENGINE_NOT_COMPILED
IARRAY_ERR_EVAL_ENGINE_OUT_OF_RANGE  = ciarray.IARRAY_ERR_EVAL_ENGINE_OUT_OF_RANGE


cdef set_storage(storage, ciarray.iarray_storage_t *cstore):
    cstore.enforce_frame = storage.enforce_frame
    if storage.plainbuffer:
        cstore.backend = ciarray.IARRAY_STORAGE_PLAINBUFFER
    else:
        cstore.backend = ciarray.IARRAY_STORAGE_BLOSC
        for i in range(len(storage.chunkshape)):
            cstore.chunkshape[i] = storage.chunkshape[i]
            cstore.blockshape[i] = storage.blockshape[i]

    if storage.filename is not None:
        filename = storage.filename.encode("utf-8") if isinstance(storage.filename, str) else storage.filename
        cstore.filename = filename
    else:
        cstore.filename = NULL


cdef class ReadBlockIter:
    cdef ciarray.iarray_iter_read_block_t *ia_read_iter
    cdef ciarray.iarray_iter_read_block_value_t ia_block_val
    cdef Container container
    cdef int dtype
    cdef int flag
    cdef object Info

    def __cinit__(self, container, block):
        self.container = container
        cdef ciarray.int64_t block_[ciarray.IARRAY_DIMENSION_MAX]
        if block is None:
            block = container.chunkshape
        for i in range(len(block)):
            block_[i] = block[i]

        iarray_check(ciarray.iarray_iter_read_block_new(self.container.context.ia_ctx, &self.ia_read_iter, self.container.ia_container, block_, &self.ia_block_val, False))
        if self.container.dtype == np.float64:
            self.dtype = 0
        else:
            self.dtype = 1
        self.Info = namedtuple('Info', 'index elemindex nblock shape size slice')

    def __dealloc__(self):
        ciarray.iarray_iter_read_block_free(&self.ia_read_iter)

    def __iter__(self):
        return self

    def __next__(self):
        if ciarray.iarray_iter_read_block_has_next(self.ia_read_iter) != 0:
            raise StopIteration

        iarray_check(ciarray.iarray_iter_read_block_next(self.ia_read_iter, NULL, 0))
        shape = tuple(self.ia_block_val.block_shape[i] for i in range(self.container.ndim))
        size = np.prod(shape)
        if self.dtype == 0:
            view = <np.float64_t[:size]> self.ia_block_val.block_pointer
        else:
            view = <np.float32_t[:size]> self.ia_block_val.block_pointer
        a = np.asarray(view)

        elem_index = tuple(self.ia_block_val.elem_index[i] for i in range(self.container.ndim))
        index = tuple(self.ia_block_val.block_index[i] for i in range(self.container.ndim))
        nblock = self.ia_block_val.nblock

        slice_ = tuple([slice(i, i + s) for i, s in zip(elem_index, shape)])
        info = self.Info(index=index, elemindex=elem_index, nblock=nblock, shape=shape,
                         size=size, slice=slice_)
        return info, a.reshape(shape)


cdef class WriteBlockIter:
    cdef ciarray.iarray_iter_write_block_t *ia_write_iter
    cdef ciarray.iarray_iter_write_block_value_t ia_block_val
    cdef Container container
    cdef int dtype
    cdef int flag
    cdef object Info

    def __cinit__(self, c, block=None):
        self.container = c
        cdef ciarray.int64_t block_[ciarray.IARRAY_DIMENSION_MAX]
        if block is None:
            # The block for iteration has always be provided
            block = c.chunkshape
        for i in range(len(block)):
            block_[i] = block[i]
        iarray_check(ciarray.iarray_iter_write_block_new(self.container.context.ia_ctx,
                                                         &self.ia_write_iter,
                                                         self.container.ia_container,
                                                         block_,
                                                         &self.ia_block_val,
                                                         False))

        if self.container.dtype == np.float64:
            self.dtype = 0
        else:
            self.dtype = 1
        self.Info = namedtuple('Info', 'index elemindex nblock shape size')

    def __dealloc__(self):
        ciarray.iarray_iter_write_block_free(&self.ia_write_iter)

    def __iter__(self):
        return self

    def __next__(self):
        if ciarray.iarray_iter_write_block_has_next(self.ia_write_iter) != 0:
            raise StopIteration

        iarray_check(ciarray.iarray_iter_write_block_next(self.ia_write_iter, NULL, 0))
        shape = tuple(self.ia_block_val.block_shape[i] for i in range(self.container.ndim))
        size = np.prod(shape)
        if self.dtype == 0:
            view = <np.float64_t[:size]> self.ia_block_val.block_pointer
        else:
            view = <np.float32_t[:size]> self.ia_block_val.block_pointer
        a = np.asarray(view)

        elem_index = tuple(self.ia_block_val.elem_index[i] for i in range(self.container.ndim))
        index = tuple(self.ia_block_val.block_index[i] for i in range(self.container.ndim))
        nblock = self.ia_block_val.nblock

        info = self.Info(index=index, elemindex=elem_index, nblock=nblock, shape=shape, size=size)
        return info, a.reshape(shape)


cdef class IArrayInit:
    def __cinit__(self):
        iarray_check(ciarray.iarray_init())

    def __dealloc__(self):
        ciarray.iarray_destroy()


cdef class Config:
    cdef ciarray.iarray_config_t config

    def __init__(self, compression_codec, compression_level, use_dict, filters,
                 max_num_threads, fp_mantissa_bits, eval_method):
        self.config.compression_codec = compression_codec.value
        self.config.compression_level = compression_level
        self.config.use_dict = 1 if use_dict else 0
        cdef int filter_flags = 0
        # TODO: filters are really a pipeline, and here we are just ORing them, which is tricky.
        # This should be fixed (probably at C iArray level and then propagating the change here).
        # At any rate, `filters` should be a list for displaying purposes in high level Config().
        for f in filters:
            filter_flags |= f.value
        if fp_mantissa_bits > 0:
            filter_flags |= ia.Filters.TRUNC_PREC.value
        self.config.filter_flags = filter_flags

        if eval_method == ia.Eval.AUTO:
            method = ciarray.IARRAY_EVAL_METHOD_AUTO
        elif eval_method == ia.Eval.ITERBLOSC:
            method = ciarray.IARRAY_EVAL_METHOD_ITERBLOSC
        elif eval_method == ia.Eval.ITERCHUNK:
            method = ciarray.IARRAY_EVAL_METHOD_ITERCHUNK
        else:
            raise ValueError("eval_method method not recognized:", eval_method)

        self.config.eval_method = method
        self.config.max_num_threads = max_num_threads
        self.config.fp_mantissa_bits = fp_mantissa_bits

    def _to_dict(self):
        return <object> self.config


cdef class Context:
    cdef ciarray.iarray_context_t *ia_ctx

    def __init__(self, cfg):
        cdef ciarray.iarray_config_t cfg_ = cfg._to_dict()
        iarray_check(ciarray.iarray_context_new(&cfg_, &self.ia_ctx))

    def __dealloc__(self):
        ciarray.iarray_context_free(&self.ia_ctx)

    def to_capsule(self):
        return PyCapsule_New(self.ia_ctx, "iarray_context_t*", NULL)


cdef class IaDTShape:
    cdef ciarray.iarray_dtshape_t ia_dtshape

    def __cinit__(self, dtshape):
        self.ia_dtshape.ndim = len(dtshape.shape)
        if dtshape.dtype == np.float64:
            self.ia_dtshape.dtype = ciarray.IARRAY_DATA_TYPE_DOUBLE
        elif dtshape.dtype == np.float32:
            self.ia_dtshape.dtype = ciarray.IARRAY_DATA_TYPE_FLOAT
        for i in range(len(dtshape.shape)):
            self.ia_dtshape.shape[i] = dtshape.shape[i]

    cdef to_dict(self):
        return <object> self.ia_dtshape

    @property
    def ndim(self):
        return self.ia_dtshape.ndim

    @property
    def dtype(self):
        dtype = [np.float64, np.float32]
        return dtype[self.ia_dtshape.dtype]

    @property
    def shape(self):
        shape = []
        for i in range(self.ndim):
            shape.append(self.ia_dtshape.shape[i])
        return tuple(shape)

    def __str__(self):
        return self.ia_dtshape


cdef class RandomContext:
    cdef ciarray.iarray_random_ctx_t *random_ctx
    cdef Context context

    def __init__(self, ctx, seed, rng):
        self.context = ctx
        cdef ciarray.iarray_random_ctx_t* r_ctx
        if rng == ia.RandomGen.MERSENNE_TWISTER:
            iarray_check(ciarray.iarray_random_ctx_new(self.context.ia_ctx, seed, ciarray.IARRAY_RANDOM_RNG_MERSENNE_TWISTER, &r_ctx))
        elif rng == ia.RandomGen.SOBOL:
            iarray_check(ciarray.iarray_random_ctx_new(self.context.ia_ctx, seed, ciarray.IARRAY_RANDOM_RNG_SOBOL, &r_ctx))
        else:
            raise ValueError("Random generator unknown")
        self.random_ctx = r_ctx

    def __dealloc__(self):
        if self.context is not None and self.context.ia_ctx != NULL:
            ciarray.iarray_random_ctx_free(self.context.ia_ctx, &self.random_ctx)
            self.context = None

    def to_capsule(self):
        return PyCapsule_New(self.random_ctx, "iarray_random_ctx_t*", NULL)


cdef class Container:
    cdef ciarray.iarray_container_t *ia_container
    cdef Context context

    def __init__(self, ctx, c):
        if ctx is None:
            raise ValueError("You must pass a context to the Container constructor")
        if c is None:
            raise ValueError("You must pass a Capsule to the C container struct of the Container constructor")
        self.context = ctx
        cdef ciarray.iarray_container_t* c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
            c, "iarray_container_t*")
        self.ia_container = c_

    def __dealloc__(self):
        if self.context is not None and self.context.ia_ctx != NULL:
            ciarray.iarray_container_free(self.context.ia_ctx, &self.ia_container)
            self.context = None

    def to_capsule(self):
        return PyCapsule_New(self.ia_container, "iarray_container_t*", NULL)

    @property
    def ndim(self):
        """Number of array dimensions."""
        cdef ciarray.iarray_dtshape_t dtshape
        iarray_check(ciarray.iarray_get_dtshape(self.context.ia_ctx, self.ia_container, &dtshape))
        return dtshape.ndim

    @property
    def shape(self):
        """Tuple of array dimensions."""
        cdef ciarray.iarray_dtshape_t dtshape
        iarray_check(ciarray.iarray_get_dtshape(self.context.ia_ctx, self.ia_container, &dtshape))
        shape = [dtshape.shape[i] for i in range(self.ndim)]
        return tuple(shape)

    @property
    def is_plainbuffer(self):
        """bool indicating if the container is based on a plainbuffer or not"""
        cdef ciarray.iarray_storage_t storage
        iarray_check(ciarray.iarray_get_storage(self.context.ia_ctx, self.ia_container, &storage))
        if storage.backend == ciarray.IARRAY_STORAGE_PLAINBUFFER:
            return True
        else:
            return False

    @property
    def chunkshape(self):
        """Tuple of chunk dimensions."""
        cdef ciarray.iarray_storage_t storage
        iarray_check(ciarray.iarray_get_storage(self.context.ia_ctx, self.ia_container, &storage))
        if storage.backend == ciarray.IARRAY_STORAGE_PLAINBUFFER or self.is_view():
            return None
        chunkshape = [storage.chunkshape[i] for i in range(self.ndim)]
        return tuple(chunkshape)

    @property
    def blockshape(self):
        """Tuple of block dimensions."""
        cdef ciarray.iarray_storage_t storage
        iarray_check(ciarray.iarray_get_storage(self.context.ia_ctx, self.ia_container, &storage))
        if storage.backend == ciarray.IARRAY_STORAGE_PLAINBUFFER or self.is_view():
            return None
        blockshape = [storage.blockshape[i] for i in range(self.ndim)]
        return tuple(blockshape)

    @property
    def dtype(self):
        """Data-type of the array’s elements."""
        dtype = [np.float64, np.float32]
        cdef ciarray.iarray_dtshape_t dtshape
        iarray_check(ciarray.iarray_get_dtshape(self.context.ia_ctx, self.ia_container, &dtshape))
        return dtype[dtshape.dtype]

    @property
    def dtshape(self):
        """The :py:obj:`DTShape` of the array."""
        return ia.DTShape(self.shape, self.dtype)

    @property
    def cratio(self):
        """Array compression ratio"""
        cdef ciarray.int64_t nbytes, cbytes
        iarray_check(ciarray.iarray_container_info(self.ia_container, &nbytes, &cbytes))
        return <double>nbytes / <double>cbytes

    def __getitem__(self, key):
        # key has been massaged already
        start, stop, squeeze_mask = key
        return get_slice(self.context, self, start, stop, squeeze_mask, True, None)

    def is_view(self):
        cdef ciarray.bool view
        iarray_check(ciarray.iarray_is_view(self.context.ia_ctx, self.ia_container, &view))
        return view

cdef class Expression:
    cdef object expression
    cdef ciarray.iarray_expression_t *ia_expr
    cdef Context context

    def __init__(self, cfg):
        self.context = Context(cfg)
        cdef ciarray.iarray_expression_t* e
        iarray_check(ciarray.iarray_expr_new(self.context.ia_ctx, &e))
        self.ia_expr = e
        self.expression = None
        self.storage = None
        self.dtshape = None

    def __dealloc__(self):
        if self.context is not None and self.context.ia_ctx != NULL:
            ciarray.iarray_expr_free(self.context.ia_ctx, &self.ia_expr)
            self.context = None

    def bind(self, var, c):
        var2 = var.encode("utf-8") if isinstance(var, str) else var
        cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
            c.to_capsule(), "iarray_container_t*")
        iarray_check(ciarray.iarray_expr_bind(self.ia_expr, var2, c_))

    def bind_out_properties(self, dtshape, storage):
        dtshape = IaDTShape(dtshape).to_dict()
        cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

        cdef ciarray.iarray_storage_t store_
        set_storage(storage, &store_)

        iarray_check(ciarray.iarray_expr_bind_out_properties(self.ia_expr, &dtshape_, &store_))
        self.dtshape = dtshape
        self.storage = storage

    def compile(self, expr):
        # Try to support all the ufuncs in numpy
        ufunc_repls = {
            "arcsin": "asin",
            "arccos": "acos",
            "arctan": "atan",
            "arctan2": "atan2",
            "power": "pow",
            }
        for ufunc in ufunc_repls.keys():
            if ufunc in expr:
                expr = expr.replace(ufunc, ufunc_repls[ufunc])
        expr = expr.encode("utf-8") if isinstance(expr, str) else expr
        iarray_check(ciarray.iarray_expr_compile(self.ia_expr, expr))
        self.expression = expr

    def compile_bc(self, bc, name):
        name = name.encode()
        iarray_check(ciarray.iarray_expr_compile_udf(self.ia_expr, len(bc), bc, name))
        self.expression = "user_defined_function"

    def compile_udf(self, func):
        self.compile_bc(func.bc, func.name)

    def eval(self):
        cdef ciarray.iarray_container_t *c;
        iarray_check(ciarray.iarray_eval(self.ia_expr, &c))
        c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
        return ia.IArray(self.context, c_c)

#
# Iarray container constructors
#

def copy(cfg, src, view=False):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    cdef ciarray.iarray_container_t *src_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(
        src.to_capsule(), "iarray_container_t*")

    iarray_check(ciarray.iarray_copy(ctx_, src_, view, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def empty(cfg, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_container_new(ctx_, &dtshape_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def arange(cfg, slice_, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    start, stop, step = slice_.start, slice_.stop, slice_.step

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_arange(ctx_, &dtshape_, start, stop, step, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def linspace(cfg, start, stop, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_linspace(ctx_, &dtshape_, start, stop, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def zeros(cfg, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_zeros(ctx_, &dtshape_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def ones(cfg, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_ones(ctx_, &dtshape_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def full(cfg, fill_value, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    if dtshape["dtype"] == ciarray.IARRAY_DATA_TYPE_DOUBLE:
        iarray_check(ciarray.iarray_fill_double(ctx_, &dtshape_, fill_value, &store_, flags, &c))
    else:
        iarray_check(ciarray.iarray_fill_float(ctx_, &dtshape_, fill_value, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def save(cfg, c, filename):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
        c.to_capsule(), "iarray_container_t*")
    filename = filename.encode("utf-8") if isinstance(filename, str) else filename

    iarray_check(ciarray.iarray_container_save(ctx_, c_, filename))


def load(cfg, filename, load_in_mem=False):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    filename = filename.encode("utf-8") if isinstance(filename, str) else filename

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_container_load(ctx_, filename, load_in_mem, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def get_slice(ctx, data, start, stop, squeeze_mask, view, storage):
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *data_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
        data.to_capsule(), "iarray_container_t*")

    shape = [sp%s - st%s for sp, st, s in zip(stop, start, data.shape)]
    dtshape = ia.DTShape(shape, data.dtype)

    flags = 0

    cdef ciarray.int64_t start_[ciarray.IARRAY_DIMENSION_MAX]
    cdef ciarray.int64_t stop_[ciarray.IARRAY_DIMENSION_MAX]

    for i in range(len(start)):
        start_[i] = start[i]
        stop_[i] = stop[i]

    cdef ciarray.iarray_storage_t store_
    cdef ciarray.iarray_container_t *c
    if view:
        iarray_check(ciarray.iarray_get_slice(ctx_, data_, start_, stop_, view, NULL, flags, &c))
    else:
        set_storage(storage, &store_)
        iarray_check(ciarray.iarray_get_slice(ctx_, data_, start_, stop_, view, &store_, flags, &c))

    cdef ciarray.bool squeeze_mask_[ciarray.IARRAY_DIMENSION_MAX]
    for i in range(data.ndim):
        squeeze_mask_[i] = squeeze_mask[i]

    iarray_check(ciarray.iarray_squeeze_index(ctx_, c, squeeze_mask_))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)

    b =  ia.IArray(ctx, c_c)
    if b.ndim == 0:
        return float(ia.iarray2numpy(b))

    return b


def numpy2iarray(cfg, a, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    buffer_size = a.size * np.dtype(a.dtype).itemsize

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_from_buffer(ctx_, &dtshape_, np.PyArray_DATA(a), buffer_size, &store_, flags, &c))

    c_c =  PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def iarray2numpy(cfg, c):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
        c.to_capsule(), "iarray_container_t*")


    cdef ciarray.iarray_dtshape_t dtshape
    iarray_check(ciarray.iarray_get_dtshape(ctx_, c_, &dtshape))
    shape = []
    for i in range(dtshape.ndim):
        shape.append(dtshape.shape[i])
    size = np.prod(shape, dtype=np.int64)

    dtype = np.float64 if dtshape.dtype == ciarray.IARRAY_DATA_TYPE_DOUBLE else np.float32
    if ciarray.iarray_is_empty(c_):
        # Return an empty array.  Another possibility would be to raise an exception here?  Let's wait for a use case...
        return np.empty(size, dtype=dtype).reshape(shape)

    a = np.zeros(size, dtype=dtype).reshape(shape)
    iarray_check(ciarray.iarray_to_buffer(ctx_, c_, np.PyArray_DATA(a), size * sizeof(dtype)))
    return a


#
# Random functions
#

def random_rand(cfg, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_rand(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_randn(cfg, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_randn(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_beta(cfg, alpha, beta, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtshape.dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_ALPHA, alpha)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_ALPHA, alpha)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_beta(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_lognormal(cfg, mu, sigma, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtshape.dtype == np.float64:
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu)
        ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma)
    else:
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu)
        ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma)

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_lognormal(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_exponential(cfg, beta, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtshape.dtype == np.float64:
        iarray_check(ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta))
    else:
        iarray_check(ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_exponential(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_uniform(cfg, a, b, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtshape.dtype == np.float64:
        iarray_check(ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_A, a))
        iarray_check(ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_B, b))
    else:
        iarray_check(ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_A, a))
        iarray_check(ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_B, b))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_uniform(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_normal(cfg, mu, sigma, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtshape.dtype == np.float64:
        iarray_check(ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu))
        iarray_check(ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma))
    else:
        iarray_check(ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu))
        iarray_check(ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_normal(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_bernoulli(cfg, p, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtshape.dtype == np.float64:
        iarray_check(ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p))
    else:
        iarray_check(ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_bernoulli(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_binomial(cfg, m, p, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtshape.dtype == np.float64:
        iarray_check(ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p))
        iarray_check(ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_M, m))
    else:
        iarray_check(ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p))
        iarray_check(ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_M, m))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_binomial(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_poisson(cfg, l, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(), "iarray_random_ctx_t*")

    if dtshape.dtype == np.float64:
        iarray_check(ciarray.iarray_random_dist_set_param_double(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_LAMBDA, l))
    else:
        iarray_check(ciarray.iarray_random_dist_set_param_float(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_LAMBDA, l))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    flags = 0 if cfg.storage.filename is None else ciarray.IARRAY_CONTAINER_PERSIST

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_poisson(ctx_, &dtshape_, r_ctx_, &store_, flags, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_kstest(cfg, a, b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.bool res;
    iarray_check(ciarray.iarray_random_kstest(ctx_, a_, b_, &res))
    return res


def matmul(cfg, a, b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg.storage, &store_)

    iarray_check(ciarray.iarray_linalg_matmul(ctx_, a_, b_, &store_, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def transpose(cfg, a):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    iarray_check(ciarray.iarray_linalg_transpose(ctx_, a_, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


# Reductions

reduce_to_c = {
    ia.Reduce.MAX: ciarray.IARRAY_REDUCE_MAX,
    ia.Reduce.MIN: ciarray.IARRAY_REDUCE_MIN,
    ia.Reduce.SUM: ciarray.IARRAY_REDUCE_SUM,
    ia.Reduce.PROD: ciarray.IARRAY_REDUCE_PROD,
    ia.Reduce.MEAN: ciarray.IARRAY_REDUCE_MEAN,
          }

def reduce(cfg, a, method, axis):
    ctx = Context(cfg)

    cdef ciarray.iarray_reduce_func_t func = reduce_to_c[method]
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    iarray_check(ciarray.iarray_reduce(ctx_, a_, func, axis, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def reduce_multi(cfg, a, method, axis):
    ctx = Context(cfg)

    cdef ciarray.iarray_reduce_func_t func = reduce_to_c[method]
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(), "iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    cdef ciarray.int8_t axis_[ciarray.IARRAY_DIMENSION_MAX]
    for i, ax in enumerate(axis):
        axis_[i] = ax

    iarray_check(ciarray.iarray_reduce_multi(ctx_, a_, func, len(axis), axis_, &c))

    c_c = PyCapsule_New(c, "iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def get_ncores(max_ncores):
    cdef int ncores = 1
    try:
        iarray_check(ciarray.iarray_get_ncores(&ncores, max_ncores))
    except IArrayError:
        # In case of error, return a negative value
        return -1
    return ncores


def partition_advice(dtshape, min_chunksize, max_chunksize, min_blocksize, max_blocksize, cfg):
    _dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> _dtshape

    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(), "iarray_context_t*")

    # Create a storage struct and initialize it.  Do we really need a store for this (maybe a frame info)?
    cdef ciarray.iarray_storage_t store
    store.backend = ciarray.IARRAY_STORAGE_BLOSC
    store.enforce_frame = False
    # Ask for the actual advice
    try:
        iarray_check(ciarray.iarray_partition_advice(ctx_, &dtshape_, &store,
                                        min_chunksize, max_chunksize, min_blocksize, max_blocksize))
    except:
        return None, None

    # Extract the shapes and return them as tuples
    chunkshape = tuple(store.chunkshape[i] for i in range(len(dtshape.shape)))
    blockshape = tuple(store.blockshape[i] for i in range(len(dtshape.shape)))
    return chunkshape, blockshape

#
# TODO: the next functions are just for benchmarking purposes and should be moved to its own extension
#
cimport numpy as cnp
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def poly_cython(xa):
    shape = xa.shape
    cdef np.ndarray[cnp.npy_float64] y = np.empty(xa.shape, xa.dtype).flatten()
    cdef np.ndarray[cnp.npy_float64] x = xa.flatten()
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
    cdef np.ndarray[cnp.npy_float64] y = np.empty(xa.shape, xa.dtype).flatten()
    cdef np.ndarray[cnp.npy_float64] x = xa.flatten()
    poly_nogil(&x[0], &y[0], len(x))
    return y.reshape(shape)

# TODO: End of the benchmarking code
