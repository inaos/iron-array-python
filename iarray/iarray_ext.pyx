# Hey Cython, this is Python 3!
# cython: language_level=3

###########################################################################################
# Copyright ironArray SL 2021.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of ironArray SL
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

from collections import namedtuple

import msgpack

from . cimport ciarray_ext as ciarray
from .ciarray_ext cimport int64_t

import numpy as np
cimport numpy as np
import zarr
import s3fs
import cython
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import iarray as ia
from iarray import udf

from cpython cimport (
    PyObject_GetBuffer,
    PyBuffer_Release,
    PyBUF_SIMPLE,
)

# dtype conversion tables: udf <-> iarray
udf2ia_dtype = {
    "f64": ciarray.IARRAY_DATA_TYPE_DOUBLE,
    "f32": ciarray.IARRAY_DATA_TYPE_FLOAT,
    "i64": ciarray.IARRAY_DATA_TYPE_INT64,
    "i32": ciarray.IARRAY_DATA_TYPE_INT32,
    "i16": ciarray.IARRAY_DATA_TYPE_INT16,
    "i8": ciarray.IARRAY_DATA_TYPE_INT8,
    "u64": ciarray.IARRAY_DATA_TYPE_UINT64,
    "u32": ciarray.IARRAY_DATA_TYPE_UINT32,
    "u16": ciarray.IARRAY_DATA_TYPE_UINT16,
    "u8": ciarray.IARRAY_DATA_TYPE_UINT8,
    "bool": ciarray.IARRAY_DATA_TYPE_BOOL,
}
ia2udf_dtype = {v: k for k, v in udf2ia_dtype.items()}

# dtype conversion tables: numpy <-> iarray
np2ia_dtype = {
    np.float64: ciarray.IARRAY_DATA_TYPE_DOUBLE,
    np.float32: ciarray.IARRAY_DATA_TYPE_FLOAT,
    np.int64: ciarray.IARRAY_DATA_TYPE_INT64,
    np.int32: ciarray.IARRAY_DATA_TYPE_INT32,
    np.int16: ciarray.IARRAY_DATA_TYPE_INT16,
    np.int8: ciarray.IARRAY_DATA_TYPE_INT8,
    np.uint64: ciarray.IARRAY_DATA_TYPE_UINT64,
    np.uint32: ciarray.IARRAY_DATA_TYPE_UINT32,
    np.uint16: ciarray.IARRAY_DATA_TYPE_UINT16,
    np.uint8: ciarray.IARRAY_DATA_TYPE_UINT8,
    np.bool_: ciarray.IARRAY_DATA_TYPE_BOOL,
}
ia2np_dtype = {v: k for k, v in np2ia_dtype.items()}

# datetime64 and timedelta64
dtype_str2dtype = {bo + l + '[' + size + ']': np.int64 for bo in ['<', '>'] for l in ['M8', 'm8']
                   for size in ['Y', 'M', 'D', 'h', 's', 'ms', 'us', 'μs', 'ns', 'ps', 'fs', 'as']}
# NaT (not a time)
dtype_str2dtype.update({bo + l: np.int64 for bo in ['<', '>'] for l in ['M8', 'm8']})
# integers
dtype_str2dtype.update({bo + 'i' + size: dtype for bo in ['<', '>']
                  for (size, dtype) in [('1', np.int8), ('2', np.int16), ('4', np.int32), ('8', np.int64)]})
# unsigned integers
dtype_str2dtype.update({bo + 'u' + size: dtype for bo in ['<', '>']
                  for (size, dtype) in [('1', np.uint8), ('2', np.uint16), ('4', np.uint32), ('8', np.uint64)]})
# floats
dtype_str2dtype.update({bo + 'f' + size: dtype for bo in ['<', '>']
                  for (size, dtype) in [('2', np.int16), ('4', np.float32), ('8', np.float64)]})
# booleans
dtype_str2dtype.update({'|b1': np.bool_, '|u1': np.bool_})


def f_np2ia_dtype(dtype):
    if type(dtype) != type:
        return np2ia_dtype[dtype.type]
    return np2ia_dtype[dtype]


def compress_squeeze(data, selectors):
    return tuple(d for d, s in zip(data, selectors) if not s)


class IArrayError(Exception):
    pass


def iarray_check(error):
    if error != 0:
        raise IArrayError(str(ciarray.iarray_err_strerror(error)))


IARRAY_ERR_EVAL_ENGINE_FAILED = ciarray.IARRAY_ERR_EVAL_ENGINE_FAILED
IARRAY_ERR_EVAL_ENGINE_NOT_COMPILED = ciarray.IARRAY_ERR_EVAL_ENGINE_NOT_COMPILED
IARRAY_ERR_EVAL_ENGINE_OUT_OF_RANGE  = ciarray.IARRAY_ERR_EVAL_ENGINE_OUT_OF_RANGE


cdef set_storage(cfg, ciarray.iarray_storage_t *cstore):
    cstore.contiguous = cfg.contiguous
    for i in range(len(cfg.chunks)):
        cstore.chunkshape[i] = cfg.chunks[i]
        cstore.blockshape[i] = cfg.blocks[i]

    if cfg.urlpath is not None:
        urlpath = cfg.urlpath.encode("utf-8") if isinstance(cfg.urlpath, str) else cfg.urlpath
        cstore.urlpath = cfg.urlpath
    else:
        cstore.urlpath = NULL


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
            block = container.chunks
        for i in range(len(block)):
            block_[i] = block[i]

        iarray_check(ciarray.iarray_iter_read_block_new(self.container.context.ia_ctx, &self.ia_read_iter,
                                                        self.container.ia_container, block_, &self.ia_block_val, False))
        self.dtype = f_np2ia_dtype(self.container.dtype)
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
        if self.dtype == ciarray.IARRAY_DATA_TYPE_DOUBLE:
            view = <np.float64_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_FLOAT:
            view = <np.float32_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_INT64:
            view = <np.int64_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_INT32:
            view = <np.int32_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_INT16:
            view = <np.int16_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_INT8:
            view = <np.int8_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_UINT64:
            view = <np.uint64_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_UINT32:
            view = <np.uint32_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_UINT16:
            view = <np.uint16_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_UINT8:
            view = <np.uint8_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_BOOL:
            view = <ciarray.bool[:size]> self.ia_block_val.block_pointer
        a = np.asarray(view)
        if self.container.np_dtype is not None:
            a = a.astype(dtype=self.container.np_dtype)

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
            block = c.chunks
        for i in range(len(block)):
            block_[i] = block[i]

        # Check that we are not inadvertently overwriting anything
        ia._check_access_mode(self.container.urlpath, self.container.mode, True)

        iarray_check(ciarray.iarray_iter_write_block_new(self.container.context.ia_ctx,
                                                         &self.ia_write_iter,
                                                         self.container.ia_container,
                                                         block_,
                                                         &self.ia_block_val,
                                                         False))

        self.dtype = f_np2ia_dtype(self.container.dtype)
        self.Info = namedtuple('Info', 'index elemindex nblock shape size')

    def __dealloc__(self):
        ciarray.iarray_iter_write_block_free(&self.ia_write_iter)

    def __iter__(self):
        return self

    def __next__(self):
        if ciarray.iarray_iter_write_block_has_next(self.ia_write_iter) != 0:
            raise StopIteration

        # Check that we are not inadvertently overwriting anything
        ia._check_access_mode(self.container.urlpath, self.container.mode, True)

        iarray_check(ciarray.iarray_iter_write_block_next(self.ia_write_iter, NULL, 0))
        shape = tuple(self.ia_block_val.block_shape[i] for i in range(self.container.ndim))
        size = np.prod(shape)
        if self.dtype == ciarray.IARRAY_DATA_TYPE_DOUBLE:
            view = <np.float64_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_FLOAT:
            view = <np.float32_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_INT64:
            view = <np.int64_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_INT32:
            view = <np.int32_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_INT16:
            view = <np.int16_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_INT8:
            view = <np.int8_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_UINT64:
            view = <np.uint64_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_UINT32:
            view = <np.uint32_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_UINT16:
            view = <np.uint16_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_UINT8:
            view = <np.uint8_t[:size]> self.ia_block_val.block_pointer
        elif self.dtype == ciarray.IARRAY_DATA_TYPE_BOOL:
            view = <ciarray.bool[:size]> self.ia_block_val.block_pointer
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

    def __init__(self, compression_codec, compression_meta, compression_level, compression_favor,
                 use_dict, filters, max_num_threads, fp_mantissa_bits, eval_method, btune, split_mode):
        self.config.compression_codec = compression_codec.value
        # Avoid error in case compression_meta < 0
        cdef ciarray.uint8_t compression_meta_
        if compression_meta is not None:
            compression_meta_ = <ciarray.int8_t> compression_meta
            self.config.compression_meta = compression_meta_
        self.config.compression_level = compression_level
        self.config.compression_favor = compression_favor.value
        self.config.use_dict = 1 if use_dict else 0
        cdef int filter_flags = 0
        # TODO: filters are really a pipeline, and here we are just ORing them, which is tricky.
        # This should be fixed (probably at C iArray level and then propagating the change here).
        # At any rate, `filters` should be a list for displaying purposes in high level Config().
        for f in filters:
            filter_flags |= f.value
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
        self.config.btune = btune
        self.config.splitmode = split_mode.value

    def _to_dict(self):
        return <object> self.config


cdef class Context:
    cdef ciarray.iarray_context_t *ia_ctx
    cdef public object cfg

    def __init__(self, cfg):
        cdef ciarray.iarray_config_t cfg_ = cfg._to_dict()
        # Set default contiguous correctly
        if cfg.contiguous is None:
            cfg.contiguous = False
        iarray_check(ciarray.iarray_context_new(&cfg_, &self.ia_ctx))
        self.cfg = cfg

    def __dealloc__(self):
        ciarray.iarray_context_free(&self.ia_ctx)

    def to_capsule(self):
        return PyCapsule_New(self.ia_ctx, <char*>"iarray_context_t*", NULL)


cdef class IaDTShape:
    cdef ciarray.iarray_dtshape_t ia_dtshape

    def __cinit__(self, dtshape):
        self.ia_dtshape.ndim = len(dtshape.shape)
        self.ia_dtshape.dtype = f_np2ia_dtype(dtshape.dtype)
        self.ia_dtshape.dtype_size = np.dtype(dtshape.dtype).itemsize
        for i in range(len(dtshape.shape)):
            self.ia_dtshape.shape[i] = dtshape.shape[i]

    cdef to_dict(self):
        return <object> self.ia_dtshape

    @property
    def ndim(self):
        return self.ia_dtshape.ndim

    @property
    def dtype(self):
        return ia2np_dtype[self.ia_dtshape.dtype]

    @property
    def dtype_size(self):
        return self.ia_dtshape.dtype_size

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
        if rng == ia.RandomGen.MRG32K3A:
            iarray_check(ciarray.iarray_random_ctx_new(self.context.ia_ctx, seed, ciarray.IARRAY_RANDOM_RNG_MRG32K3A, &r_ctx))
        else:
            raise ValueError("Random generator unknown")
        self.random_ctx = r_ctx

    def __dealloc__(self):
        if self.context is not None and self.context.ia_ctx != NULL:
            ciarray.iarray_random_ctx_free(self.context.ia_ctx, &self.random_ctx)
            self.context = None

    def to_capsule(self):
        return PyCapsule_New(self.random_ctx, <char*>"iarray_random_ctx_t*", NULL)

def _is_s3_store(urlpath):
    if urlpath[:5] == "s3://":
        return True
    return False

cpdef _zarray_from_proxy(urlpath):
    if _is_s3_store(urlpath):
        s3 = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=urlpath, s3=s3)
        return zarr.open(store)
    else:
        return zarr.open(urlpath)

cdef class Container:
    cdef ciarray.iarray_container_t *ia_container
    cdef Context context
    cdef Py_ssize_t bp_shape[ciarray.IARRAY_DIMENSION_MAX]
    cdef Py_ssize_t bp_strides[ciarray.IARRAY_DIMENSION_MAX]
    cdef int view_count

    def __init__(self, ctx, c):
        if ctx is None:
            raise ValueError("You must pass a context to the Container constructor")
        if c is None:
            raise ValueError("You must pass a Capsule to the C container struct of the Container constructor")
        self.context = ctx
        self.ia_container = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c, <char*>"iarray_container_t*")
        self.buffer = None
        self.view_count = 0

    def __dealloc__(self):
        if self.context is not None and self.context.ia_ctx != NULL:
            # if self.view_count > 0:
            #     # TODO: set Blosc flag `cframe_avoid_free = True`
            #     pass

            ciarray.iarray_container_free(self.context.ia_ctx, &self.ia_container)
            self.context = None

    # THERE ARE COLLISIONS WITH LAZY EXPRESSIONS
    #
    # def __getbuffer__(self, Py_buffer *buffer, int flags):
    #     dtype = np.dtype(self.dtype)
    #
    #     ctx = Context(ia.Config())
    #     cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(
    #         ctx.to_capsule(),
    #         <char *> "iarray_context_t*")
    #
    #     cdef ciarray.uint8_t *cframe
    #     cdef ciarray.int64_t cframe_len
    #     cdef ciarray.bool needs_free
    #     iarray_check(ciarray.iarray_to_cframe(ctx_, self.ia_container, &cframe, &cframe_len, &needs_free))
    #
    #     self.bp_shape[0] = cframe_len
    #     self.bp_strides[0] = 1
    #
    #     buffer.buf = <char *> cframe
    #     buffer.format = 'B'  # unsigned bytes (compressed array)
    #     buffer.internal = NULL  # see References
    #     buffer.readonly = 1
    #     buffer.obj = self
    #     buffer.itemsize = 1
    #     buffer.len = cframe_len
    #     buffer.ndim = 1
    #     buffer.shape = self.bp_shape
    #     buffer.strides = self.bp_strides
    #     buffer.suboffsets = NULL
    #     if not needs_free:
    #         self.view_count += 1
    # def __releasebuffer__(self, Py_buffer *buffer):
    #     self.view_count -= 1


    def to_capsule(self):
        return PyCapsule_New(self.ia_container, <char*>"iarray_container_t*", NULL)

    def to_cframe(self):
        """ Create a cframe.

        Returns
        -------
        A bytes object containing the cframe
        """
        return get_cframe(self)

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
    def chunks(self):
        """Tuple of chunk dimensions."""
        cdef ciarray.iarray_storage_t storage
        iarray_check(ciarray.iarray_get_storage(self.context.ia_ctx, self.ia_container, &storage))
        if self.is_view:
            return None
        chunks = [storage.chunkshape[i] for i in range(self.ndim)]
        return tuple(chunks)

    @property
    def blocks(self):
        """Tuple of block dimensions."""
        cdef ciarray.iarray_storage_t storage
        iarray_check(ciarray.iarray_get_storage(self.context.ia_ctx, self.ia_container, &storage))
        if self.is_view:
            return None
        blocks = [storage.blockshape[i] for i in range(self.ndim)]
        return tuple(blocks)

    @property
    def dtype(self):
        """Data-type of the array’s elements."""
        cdef ciarray.iarray_dtshape_t dtshape
        iarray_check(ciarray.iarray_get_dtshape(self.context.ia_ctx, self.ia_container, &dtshape))
        return ia2np_dtype[dtshape.dtype]

    @property
    def np_dtype(self):
        """The array-protocol typestring of the np.dtype object to use."""
        return self.attrs["np_dtype"] if "np_dtype" in self.attrs.keys() else None

    @np_dtype.setter
    def np_dtype(self, value):
        if value is not None:
            self.attrs["np_dtype"] = np.dtype(value).str

    @property
    def dtshape(self):
        """The :py:obj:`DTShape` of the array."""
        return ia.DTShape(self.shape, self.dtype)

    @property
    def cratio(self):
        """Array compression ratio."""
        # Return zarr array values if it is a zproxy
        if "zproxy_urlpath" in self.attrs:
            urlpath = self.attrs["zproxy_urlpath"]
            z = _zarray_from_proxy(urlpath)
            return z.nbytes / z.nbytes_stored
        # It is a normal iarray
        cdef ciarray.int64_t nbytes, cbytes
        iarray_check(ciarray.iarray_container_info(self.ia_container, &nbytes, &cbytes))
        return <double>nbytes / <double>cbytes

    @property
    def cfg(self):
        return self.context.cfg

    @property
    def urlpath(self):
        return self.context.cfg.urlpath

    @property
    def mode(self):
        return self.context.cfg.mode

    def __getitem__(self, key):
        # key has been massaged already
        start, stop, squeeze_mask = key

        with ia.config(cfg=self.cfg) as cfg:
            return get_slice(cfg, self, start, stop, squeeze_mask, True, None)

    @property
    def is_view(self):
        """Whether the :ref:`IArray` is a view or not.
        """
        cdef ciarray.bool view
        iarray_check(ciarray.iarray_is_view(self.context.ia_ctx, self.ia_container, &view))
        return view


cdef class Expression:
    cdef object expression
    cdef ciarray.iarray_expression_t *ia_expr
    cdef Context context
    cdef ciarray.bool zproxy_op

    def __init__(self, cfg):
        self.cfg = cfg
        self.context = Context(cfg)
        cdef ciarray.iarray_expression_t* e
        iarray_check(
            ciarray.iarray_expr_new(self.context.ia_ctx, f_np2ia_dtype(cfg.dtype), &e)
        )
        self.ia_expr = e
        self.expression = None
        self.dtshape = None
        self.zproxy_op = False

    def __dealloc__(self):
        if self.context is not None and self.context.ia_ctx != NULL:
            ciarray.iarray_expr_free(self.context.ia_ctx, &self.ia_expr)
            self.context = None

    def bind(self, var, c):
        var2 = var.encode("utf-8") if isinstance(var, str) else var
        if "zproxy_urlpath" in c.attrs:
            # To not release the GIL when evaluating
            self.zproxy_op = True
        cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(
            c.to_capsule(), <char*>"iarray_container_t*")
        iarray_check(ciarray.iarray_expr_bind(self.ia_expr, var2, c_))

    def bind_param(self, value, type_):
        cdef ciarray.iarray_user_param_t user_param;

        if type_ is udf.float64:
            user_param.f64 = value
        elif type_ is udf.float32:
            user_param.f32 = value
        elif type_ is udf.int64:
            user_param.i64 = value
        elif type_ is udf.int32:
            user_param.i32 = value
        elif type_ is udf.bool:
            user_param.b = value

        iarray_check(
            ciarray.iarray_expr_bind_param(self.ia_expr, user_param)
        )

    def bind_out_properties(self, dtshape):
        dtshape = IaDTShape(dtshape).to_dict()
        cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

        cdef ciarray.iarray_storage_t store_
        set_storage(self.cfg, &store_)

        iarray_check(ciarray.iarray_expr_bind_out_properties(self.ia_expr, &dtshape_, &store_))
        self.dtshape = dtshape

    def compile(self, expr):
        expr = expr.encode("utf-8") if isinstance(expr, str) else expr
        iarray_check(ciarray.iarray_expr_compile(self.ia_expr, expr))
        self.expression = expr

    def compile_bc(self, bc, name):
        name = name.encode()
        cdef int bc_len = len(bc)
        iarray_check(ciarray.iarray_expr_compile_udf(self.ia_expr, bc_len, bc, name))
        self.expression = "user_defined_function"

    def compile_udf(self, func):
        self.compile_bc(func.bc, func.name)

    def eval(self):
        cdef ciarray.iarray_container_t *c;
        # Check that we are not inadvertently overwriting anything
        ia._check_access_mode(self.cfg.urlpath, self.cfg.mode)
        # Update the chunks and blocks with the correct values
        self.update_chunks_blocks()
        if self.zproxy_op:
            error = ciarray.iarray_eval(self.ia_expr, &c)
        else:
            with nogil:
                error = ciarray.iarray_eval(self.ia_expr, &c)
        iarray_check(error)
        c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
        return ia.IArray(self.context, c_c)

    def update_chunks_blocks(self):
        cdef int nvars = self.ia_expr.nvars;
        cdef ciarray.int8_t ndim = <ciarray.int8_t> self.dtshape["ndim"]
        cdef ciarray.iarray_storage_t storage_0
        cdef ciarray.iarray_storage_t storage_i

        if self.cfg.chunks is None:
            # Set blocks and chunks to the ones from the operands in case all of them are equal
            if nvars > 0:
                equal = True
                ciarray.iarray_get_storage(self.ia_expr.ctx, self.ia_expr.vars[0].c, &storage_0)
                chunks_0 = list(storage_0.chunkshape)[:ndim]
                blocks_0 = list(storage_0.blockshape)[:ndim]
                for i in range(1, nvars):
                    ciarray.iarray_get_storage(self.ia_expr.ctx, self.ia_expr.vars[i].c, &storage_i)
                    chunks_i = list(storage_i.chunkshape)[:ndim]
                    blocks_i = list(storage_i.blockshape)[:ndim]
                    if chunks_i != chunks_0 or blocks_i != blocks_0:
                        equal = False
                        break
                if equal:
                    self.ia_expr.out_store_properties.chunkshape = storage_0.chunkshape
                    self.ia_expr.out_store_properties.blockshape = storage_0.blockshape

        chunks = list(self.ia_expr.out_store_properties.chunkshape)[:ndim]
        blocks = list(self.ia_expr.out_store_properties.blockshape)[:ndim]
        self.cfg.chunks = chunks
        self.cfg.blocks = blocks


#
# Iarray container constructors
#

def copy(cfg, src):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(
        ctx.to_capsule(), <char*>"iarray_context_t*")

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    cdef ciarray.iarray_container_t *src_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(
        src.to_capsule(), <char*>"iarray_container_t*")

    if "zproxy_urlpath" in src.attrs:
        error = ciarray.iarray_copy(ctx_, src_, False, &store_, &c)
    else:
        with nogil:
            error = ciarray.iarray_copy(ctx_, src_, False, &store_, &c)
    iarray_check(error)

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    a = ia.IArray(ctx, c_c)
    a.np_dtype = cfg.np_dtype

    return a


def uninit(cfg, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_uninit(ctx_, &dtshape_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    a = ia.IArray(ctx, c_c)
    a.np_dtype = cfg.np_dtype

    return a


def arange(cfg, slice_, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    start, stop, step = slice_.start, slice_.stop, slice_.step

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)


    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_arange(ctx_, &dtshape_, start, step, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    a = ia.IArray(ctx, c_c)
    a.np_dtype = cfg.np_dtype

    return a


def linspace(cfg, start, stop, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_linspace(ctx_, &dtshape_, start, stop, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    a = ia.IArray(ctx, c_c)
    a.np_dtype = cfg.np_dtype

    return a


def zeros(cfg, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_zeros(ctx_, &dtshape_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    a = ia.IArray(ctx, c_c)
    a.np_dtype = cfg.np_dtype

    return a


def ones(cfg, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_ones(ctx_, &dtshape_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    a = ia.IArray(ctx, c_c)
    a.np_dtype = cfg.np_dtype

    return a


def full(cfg, fill_value, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    dtype = ia2np_dtype[dtshape_.dtype]
    # The ciarray.iarray_fill function requires a void pointer
    nparr = np.array([fill_value], dtype=dtype)
    cdef Py_buffer *val = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(nparr, val, PyBUF_SIMPLE)

    iarray_check(ciarray.iarray_fill(ctx_, &dtshape_, val.buf, &store_, &c))
    PyBuffer_Release(val)


    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    a = ia.IArray(ctx, c_c)
    a.np_dtype = cfg.np_dtype

    return a

cdef get_cfg_from_container(cfg, ciarray.iarray_context_t *ctx, ciarray.iarray_container_t *c, urlpath):
    cdef ciarray.iarray_config_t cfg_
    ciarray.iarray_get_cfg(ctx, c, &cfg_)

    clevel = cfg_.compression_level
    codec = ia.Codec(cfg_.compression_codec)
    zfp_meta = cfg_.compression_meta
    mantissa_bits = cfg_.fp_mantissa_bits

    filters = []
    if cfg_.filter_flags & ciarray.IARRAY_COMP_TRUNC_PREC:
        filters.append(ia.Filter.TRUNC_PREC)
    if cfg_.filter_flags & ciarray.IARRAY_COMP_DELTA:
        filters.append(ia.Filter.DELTA)
    if cfg_.filter_flags & ciarray.IARRAY_COMP_BITSHUFFLE:
        filters.append(ia.Filter.BITSHUFFLE)
    if cfg_.filter_flags & ciarray.IARRAY_COMP_SHUFFLE:
        filters.append(ia.Filter.SHUFFLE)

    cdef ciarray.iarray_dtshape_t dtshape;
    ciarray.iarray_get_dtshape(ctx, c, &dtshape)

    dtype = ia2np_dtype[dtshape.dtype]

    cdef const char *name = "np_dtype"
    cdef ciarray.bool exists
    iarray_check(ciarray.iarray_vlmeta_exists(ctx, c, name, &exists))
    cdef ciarray.iarray_metalayer_t meta
    if exists:
        iarray_check(ciarray.iarray_vlmeta_get(ctx, c, name, &meta))
        np_dtype = meta.sdata[:meta.size]
        np_dtype = msgpack.unpackb(np_dtype)
    else:
        np_dtype = None

    cdef ciarray.iarray_storage_t storage;
    ciarray.iarray_get_storage(ctx, c, &storage)

    chunks = tuple(storage.chunkshape[i] for i in range(dtshape.ndim))
    blocks = tuple(storage.blockshape[i] for i in range(dtshape.ndim))

    contiguous = storage.contiguous

    # The config params should already have been checked
    ia._defaults.check_compat = False
    c_cfg = ia.Config(
        codec=codec,
        zfp_meta=zfp_meta,
        clevel=clevel,
        filters=filters,
        fp_mantissa_bits = mantissa_bits,
        use_dict=False,
        favor=cfg.favor,
        nthreads=cfg.nthreads,
        eval_method=cfg.eval_method,
        seed=cfg.seed,
        random_gen=cfg.random_gen,
        btune=False,   # we have not used btune to load/open
        dtype=dtype,
        np_dtype=np_dtype,
        chunks=chunks,
        blocks=blocks,
        urlpath=urlpath,
        contiguous=contiguous,
        mode=cfg.mode,
    )
    return c_cfg


def load(cfg, urlpath):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    urlpath = urlpath.encode("utf-8") if isinstance(urlpath, str) else urlpath

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_container_load(ctx_, urlpath, &c))

    # Fetch config from the new container
    c_cfg = get_cfg_from_container(cfg, ctx_, c, None)

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(Context(c_cfg), c_c)


def open(cfg, urlpath):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    urlpath = urlpath.encode("utf-8") if isinstance(urlpath, str) else urlpath

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_container_open(ctx_, urlpath, &c))

    # Fetch config from the recently open container
    c_cfg = get_cfg_from_container(cfg, ctx_, c, urlpath)

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    iarr = ia.IArray(Context(c_cfg), c_c)
    if "zproxy_urlpath" in iarr.attrs:
        set_zproxy_postfilter(iarr)
    return iarr

def set_orthogonal_selection(cfg, dst, selection, ndarray):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *data_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(dst.to_capsule(),
                                                                                                <char*>"iarray_container_t*")
    ndim = dst.ndim
    interface = ndarray.__array_interface__
    cdef Py_buffer buf
    PyObject_GetBuffer(ndarray, &buf, PyBUF_SIMPLE)

    cdef ciarray.int64_t **selection_ = <ciarray.int64_t **> malloc(len(selection) * sizeof(ciarray.int64_t *))
    cdef ciarray.int64_t *selection_size_ = <ciarray.int64_t *> malloc(len(selection) * sizeof(ciarray.int64_t))

    for i, sel in enumerate(selection):
        selection_[i] = <ciarray.int64_t *> malloc(len(sel) * sizeof(ciarray.int64_t))
        selection_size_[i] = len(sel)
        for j, s in enumerate(sel):
            selection_[i][j] = s

    cdef ciarray.int64_t buffersize_ = np.dtype(dst.dtype).itemsize
    cdef ciarray.int64_t[ciarray.IARRAY_DIMENSION_MAX] buffershape_
    for i, sel in enumerate(selection):
        buffershape_[i] = len(sel)
        buffersize_ *= buffershape_[i]

    buffershape = [len(sel) for sel in selection]


    iarray_check(ciarray.iarray_set_orthogonal_selection(ctx_, data_, selection_, selection_size_,
                                     <void *> buf.buf, buffershape_, buffersize_))
    PyBuffer_Release(&buf)

    return dst


cdef get_cframe(data):
    ctx = Context(ia.Config())
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(
        ctx.to_capsule(),
        <char *> "iarray_context_t*")

    cdef ciarray.iarray_container_t *data_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(data.to_capsule(),
                                                                                                <char*>"iarray_container_t*")
    cdef ciarray.uint8_t *cframe
    cdef ciarray.int64_t cframe_len
    cdef ciarray.bool needs_free
    iarray_check(
        ciarray.iarray_to_cframe(ctx_, data_, &cframe, &cframe_len, &needs_free))
    b = bytes(memoryview(cframe[:cframe_len]))  # copy is done
    if needs_free:
        free(cframe)
    return b



def set_slice(cfg, data, start, stop, buffer):
    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode, True)

    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *data_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(data.to_capsule(),
                                                                                                <char*>"iarray_container_t*")

    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(buffer, buf, PyBUF_SIMPLE)

    cdef ciarray.int64_t start_[ciarray.IARRAY_DIMENSION_MAX]
    cdef ciarray.int64_t stop_[ciarray.IARRAY_DIMENSION_MAX]

    for i in range(len(start)):
        start_[i] = start[i]
        stop_[i] = stop[i]

    iarray_check(ciarray.iarray_set_slice_buffer(ctx_, data_, start_, stop_, buf.buf, buf.len))
    PyBuffer_Release(buf)

    return data

def get_orthogonal_selection(cfg, src, dst, selection):
    ia._check_access_mode(cfg.urlpath, cfg.mode, True)

    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(
        ctx.to_capsule(),
        <char *> "iarray_context_t*")
    cdef ciarray.iarray_container_t *data_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(
        src.to_capsule(),
        <char *> "iarray_container_t*")

    cdef ciarray.int64_t ** selection_ = <ciarray.int64_t **> malloc(
        len(selection) * sizeof(ciarray.int64_t *))
    cdef ciarray.int64_t *selection_size_ = <ciarray.int64_t *> malloc(
        len(selection) * sizeof(ciarray.int64_t))

    for i, sel in enumerate(selection):
        selection_[i] = <ciarray.int64_t *> malloc(len(sel) * sizeof(ciarray.int64_t))
        selection_size_[i] = len(sel)
        for j, s in enumerate(sel):
            selection_[i][j] = s

    cdef ciarray.int64_t buffersize_ = np.dtype(dst.dtype).itemsize
    cdef ciarray.int64_t[ciarray.IARRAY_DIMENSION_MAX] buffershape_
    for i, sel in enumerate(selection):
        buffershape_[i] = len(sel)
        buffersize_ *= buffershape_[i]

    buffershape = [len(sel) for sel in selection]

    cdef Py_buffer view
    PyObject_GetBuffer(dst, &view, PyBUF_SIMPLE)

    iarray_check(ciarray.iarray_get_orthogonal_selection(ctx_, data_, selection_, selection_size_,
                                     <void *> view.buf, buffershape_, buffersize_))
    PyBuffer_Release(&view)

    return dst


def get_slice(cfg, data, start, stop, squeeze_mask, view, storage):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *data_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(data.to_capsule(),
                                                                                                <char*>"iarray_container_t*")

    shape = [sp%s - st%s for sp, st, s in zip(stop, start, data.shape)]
    dtshape = ia.DTShape(shape, data.dtype)

    cdef ciarray.int64_t start_[ciarray.IARRAY_DIMENSION_MAX]
    cdef ciarray.int64_t stop_[ciarray.IARRAY_DIMENSION_MAX]

    for i in range(len(start)):
        start_[i] = start[i]
        stop_[i] = stop[i]

    cdef ciarray.iarray_storage_t store_
    cdef ciarray.iarray_container_t *c

    cfg = ctx.cfg
    if view:
        if cfg.blocks and cfg.chunks:
            shape = tuple(sp - st for sp, st in zip(stop, start))
            chunks = list(cfg.chunks)
            blocks = list(cfg.blocks)
            for i, s in enumerate(shape):
                if s < cfg.chunks[i]:
                    chunks[i] = s
                if chunks[i] < cfg.blocks[i]:
                    blocks[i] = chunks[i]
            cfg.chunks = compress_squeeze(chunks, squeeze_mask)
            cfg.blocks = compress_squeeze(blocks, squeeze_mask)

        iarray_check(ciarray.iarray_get_slice(ctx_, data_, start_, stop_, view, NULL, &c))
    else:
        set_storage(cfg, &store_)
        iarray_check(ciarray.iarray_get_slice(ctx_, data_, start_, stop_, view, &store_, &c))

    cdef ciarray.bool squeeze_mask_[ciarray.IARRAY_DIMENSION_MAX]
    for i in range(data.ndim):
        squeeze_mask_[i] = squeeze_mask[i]

    iarray_check(ciarray.iarray_squeeze_index(ctx_, c, squeeze_mask_))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)

    b =  ia.IArray(ctx, c_c)
    b.view_ref = data  # Keep a reference of the parent container

    return b


def get_type_view(cfg, iarray, view_dtype):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *src = <ciarray.iarray_container_t*> PyCapsule_GetPointer(iarray.to_capsule(),
                                                                                            <char*>"iarray_container_t*")
    cdef ciarray.iarray_data_type_t dtype = f_np2ia_dtype(view_dtype)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_get_type_view(ctx_, src, dtype, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)

    b =  ia.IArray(ctx, c_c)
    b.view_ref = iarray  # Keep a reference of the parent container

    return b


def resize(container, new_shape, start):
    cdef ciarray.iarray_container_t *container_
    container_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(container.to_capsule(),
                                                                     <char*>"iarray_container_t*")
    ctx = Context(container.cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                            <char*>"iarray_context_t*")
    cdef int ndim = len(new_shape)
    cdef ciarray.int64_t new_shape_[ciarray.IARRAY_DIMENSION_MAX]
    for i in range(ndim):
        new_shape_[i] = new_shape[i]
    cdef ciarray.int64_t start_[ciarray.IARRAY_DIMENSION_MAX]
    if start is None:
        iarray_check(ciarray.iarray_container_resize(ctx_, container_, new_shape_, NULL))
    else:
        for i in range(ndim):
            start_[i] = start[i]
        iarray_check(ciarray.iarray_container_resize(ctx_, container_, new_shape_, start_))


def insert(container, buffer, axis, start):
    if start is not  None:
        if start % container.chunks[axis] != 0 and start != container.shape[axis]:
            raise IndexError("Cannot insert in the middle of the chunks")

    cdef ciarray.iarray_container_t *container_
    container_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(container.to_capsule(),
                                                                     <char*>"iarray_container_t*")
    ctx = Context(container.cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                            <char*>"iarray_context_t*")
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(buffer, buf, PyBUF_SIMPLE)
    if start is None:
        start = container.shape[axis]

    row_size = np.dtype(container.dtype).itemsize
    buffershape = [0] * container.ndim
    for i in range(0, container.ndim):
        if i != axis:
            row_size *= container.shape[i]
            buffershape[i] = container.shape[i]

    if buf.len % row_size != 0:
        raise ValueError("The data length must be a multiple of the array's shape excluding the axis")

    iarray_check(ciarray.iarray_container_insert(ctx_, container_, buf.buf, buf.len, axis, start))

    PyBuffer_Release(buf)


def append(container, buffer, axis):
    cdef ciarray.iarray_container_t *container_
    container_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(container.to_capsule(),
                                                                     <char*>"iarray_container_t*")
    ctx = Context(container.cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                            <char*>"iarray_context_t*")
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(buffer, buf, PyBUF_SIMPLE)

    row_size = np.dtype(container.dtype).itemsize
    for i in range(0, container.ndim):
        if i != axis:
            row_size *= container.shape[i]
    if buf.len % row_size != 0:
        raise ValueError("The data length must be a multiple of the array's shape excluding the axis")

    iarray_check(ciarray.iarray_container_append(ctx_, container_, buf.buf, buf.len, axis))

    PyBuffer_Release(buf)


def delete(container, axis, delete_len, start):
    if start is not None:
        if (start + delete_len) != container.shape[axis]:
            if start % container.chunks[axis] != 0 or (start + delete_len) % container.chunks[axis] != 0:
                raise IndexError("Cannot delete in the middle of the chunks")

    cdef ciarray.iarray_container_t *container_
    container_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(container.to_capsule(),
                                                                     <char*>"iarray_container_t*")
    ctx = Context(container.cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                            <char*>"iarray_context_t*")
    if start is None:
        start = container.shape[axis] - delete_len

    iarray_check(ciarray.iarray_container_delete(ctx_, container_, axis, start, delete_len))


def numpy2iarray(cfg, a, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    buffer_size = a.size * np.dtype(a.dtype).itemsize

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_from_buffer(ctx_, &dtshape_, np.PyArray_DATA(a), buffer_size, &store_, &c))

    c_c =  PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    b = ia.IArray(ctx, c_c)
    b.np_dtype = cfg.np_dtype

    return b


def split(cfg, container):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(), <char *> "iarray_context_t*")

    cdef ciarray.iarray_container_t *container_
    container_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(container.to_capsule(),
                                                                     <char *> "iarray_container_t*")

    nchunks = 1
    for s, c in zip(container.shape, container.chunks):
        nchunks *= s // c if s % c == 0 else s // c + 1

    cdef ciarray.iarray_container_t **dst_ = <ciarray.iarray_container_t **> malloc(nchunks * sizeof(ciarray.iarray_container_t *))
    iarray_check(ciarray.iarray_split(ctx_, container_, dst_))

    l = []
    for i in range(nchunks):
        c_c = PyCapsule_New(dst_[i], <char *> "iarray_container_t*", NULL)
        l += [ia.IArray(ctx, c_c)]

    free(dst_)
    return l


def concatenate(cfg, l, dtshape):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(), <char *> "iarray_context_t*")

    cdef ciarray.iarray_container_t **dst_ = <ciarray.iarray_container_t **> malloc(len(l) * sizeof(ciarray.iarray_container_t *))

    for i in range(len(l)):
        dst_[i] = <ciarray.iarray_container_t *> PyCapsule_GetPointer(l[i].to_capsule(),
                                                                     <char *> "iarray_container_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)


    cdef ciarray.iarray_container_t *c_
    iarray_check(ciarray.iarray_concatenate(ctx_, dst_, &dtshape_, &store_, &c_))

    c_c = PyCapsule_New(c_, <char *> "iarray_container_t*", NULL)
    b = ia.IArray(ctx, c_c)

    return b


def from_cframe(cfg, cframe: [bytes, bytearray], copy: bool = False):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char *> "iarray_context_t*")

    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(cframe, buf, PyBUF_SIMPLE)

    cdef ciarray.iarray_container_t *c
    iarray_check(
        ciarray.iarray_from_cframe(ctx_, <ciarray.uint8_t*> <char *> buf.buf, buf.len, copy, &c))
    c_c = PyCapsule_New(c, <char *> "iarray_container_t*", NULL)
    b = ia.IArray(ctx, c_c)
    b.buffer = cframe

    b.np_dtype = cfg.np_dtype  # No idea :(

    b = ia.IArray.cast(b)

    return b


def from_chunk_index(cfg, src, shape, chunk_index):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                            <char *> "iarray_context_t*")


    cdef ciarray.iarray_container_t *src_
    src_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(src.to_capsule(),
                                                                     <char *> "iarray_container_t*")

    cdef int64_t *shape_ = <int64_t *> malloc(ciarray.IARRAY_DIMENSION_MAX * sizeof(int64_t))
    for i in range(src.ndim):
        shape_[i] = shape[i]

    cdef int64_t *chunk_indexes_ = <int64_t *> malloc(len(chunk_index) * sizeof(int64_t))
    cdef int64_t chunk_indexes_len_ = len(chunk_index)
    for i in range(len(chunk_index)):
        chunk_indexes_[i] = chunk_index[i]

    cdef ciarray.iarray_container_t *c_
    iarray_check(ciarray.iarray_from_chunk_index(ctx_, src_, shape_, chunk_indexes_, chunk_indexes_len_, &c_))
    free(shape_)
    free(chunk_indexes_)
    c_c = PyCapsule_New(c_, <char *> "iarray_container_t*", NULL)
    b = ia.IArray(ctx, c_c)

    return b


def iarray2numpy(cfg, c):
    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *c_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(c.to_capsule(),
                                                                                             <char*>"iarray_container_t*")

    cdef ciarray.iarray_dtshape_t dtshape
    iarray_check(ciarray.iarray_get_dtshape(ctx_, c_, &dtshape))
    shape = []
    for i in range(dtshape.ndim):
        shape.append(dtshape.shape[i])
    size = np.prod(shape, dtype=np.int64)

    dtype = ia2np_dtype[dtshape.dtype]
    if ciarray.iarray_is_empty(c_):
        # Return an empty array.  Another possibility would be to raise an exception here?  Let's wait for a use case...
        if c.np_dtype is not None:
            dtype = np.dtype(c.np_dtype)
        return np.empty(size, dtype=dtype).reshape(shape)

    if c.np_dtype is None:
        a = np.zeros(size, dtype=dtype).reshape(shape)
        itemsize = sizeof(dtype)
        iarray_check(ciarray.iarray_to_buffer(ctx_, c_, np.PyArray_DATA(a), size * itemsize))
        return a
    else:
        type = np.dtype(c.np_dtype)
        if dtype == dtype_str2dtype[c.np_dtype]:
            a = np.zeros(size, dtype=type).reshape(shape)
            itemsize = type.itemsize
            iarray_check(ciarray.iarray_to_buffer(ctx_, c_, np.PyArray_DATA(a), size * itemsize))
        else:
            # dtypes are incompatible, so force the cast
            a = np.zeros(size, dtype=dtype).reshape(shape)
            itemsize = np.dtype(dtype).itemsize
            iarray_check(ciarray.iarray_to_buffer(ctx_, c_, np.PyArray_DATA(a), size * itemsize))
            return a.astype(type)
        if type.str[0] == '>':
            a = a.byteswap()
        return a


#
# Random functions
#

def random_rand(cfg, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_rand(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_randn(cfg, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_randn(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_beta(cfg, alpha, beta, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_ALPHA, alpha)
    ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta)

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_beta(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_lognormal(cfg, mu, sigma, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu)
    ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma)

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_lognormal(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_exponential(cfg, beta, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    iarray_check(ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_BETA, beta))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_exponential(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_uniform(cfg, a, b, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    iarray_check(ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_A, a))
    iarray_check(ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_B, b))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_uniform(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_normal(cfg, mu, sigma, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    iarray_check(ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_MU, mu))
    iarray_check(ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_SIGMA, sigma))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_normal(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_bernoulli(cfg, p, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    iarray_check(ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_bernoulli(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_binomial(cfg, m, p, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    iarray_check(ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_P, p))
    iarray_check(ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_M, m))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_binomial(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_poisson(cfg, l, dtshape):
    ctx = Context(cfg)
    r_ctx = RandomContext(ctx, cfg.seed, cfg.random_gen)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_random_ctx_t *r_ctx_ = <ciarray.iarray_random_ctx_t*> PyCapsule_GetPointer(r_ctx.to_capsule(),
                                                                                                   <char*>"iarray_random_ctx_t*")

    iarray_check(ciarray.iarray_random_dist_set_param(r_ctx_, ciarray.IARRAY_RANDOM_DIST_PARAM_LAMBDA, l))

    dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> dtshape

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.iarray_container_t *c
    iarray_check(ciarray.iarray_random_poisson(ctx_, &dtshape_, r_ctx_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def random_kstest(cfg, a, b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.bool res;
    iarray_check(ciarray.iarray_random_kstest(ctx_, a_, b_, &res))
    return res


def matmul(cfg, a, b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    iarray_check(ciarray.iarray_linalg_matmul(ctx_, a_, b_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    d = ia.IArray(ctx, c_c)
    d.np_dtype = cfg.np_dtype

    return d


def opt_gemv(cfg, a, b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    iarray_check(ciarray.iarray_opt_gemv(ctx_, a_, b_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def opt_gemm(cfg, a, b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    iarray_check(ciarray.iarray_opt_gemm(ctx_, a_, b_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def opt_gemm_b(cfg, a, b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    iarray_check(ciarray.iarray_opt_gemm_b(ctx_, a_, b_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def opt_gemm_a(cfg, a, b):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_container_t *b_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(b.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    iarray_check(ciarray.iarray_opt_gemm_a(ctx_, a_, b_, &store_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def transpose(cfg, a):
    ctx = Context(cfg)
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    iarray_check(ciarray.iarray_linalg_transpose(ctx_, a_, &c))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    d = ia.IArray(ctx, c_c)

    return d


# Reductions

reduce_to_c = {
    ia.Reduce.MAX: ciarray.IARRAY_REDUCE_MAX,
    ia.Reduce.MIN: ciarray.IARRAY_REDUCE_MIN,
    ia.Reduce.SUM: ciarray.IARRAY_REDUCE_SUM,
    ia.Reduce.PROD: ciarray.IARRAY_REDUCE_PROD,
    ia.Reduce.MEAN: ciarray.IARRAY_REDUCE_MEAN,
    ia.Reduce.STD: ciarray.IARRAY_REDUCE_STD,
    ia.Reduce.VAR: ciarray.IARRAY_REDUCE_VAR,
    ia.Reduce.MEDIAN: ciarray.IARRAY_REDUCE_MEDIAN,
    ia.Reduce.NAN_MAX: ciarray.IARRAY_REDUCE_NAN_MAX,
    ia.Reduce.NAN_MIN: ciarray.IARRAY_REDUCE_NAN_MIN,
    ia.Reduce.NAN_SUM: ciarray.IARRAY_REDUCE_NAN_SUM,
    ia.Reduce.NAN_PROD: ciarray.IARRAY_REDUCE_NAN_PROD,
    ia.Reduce.NAN_MEAN: ciarray.IARRAY_REDUCE_NAN_MEAN,
    ia.Reduce.NAN_STD: ciarray.IARRAY_REDUCE_NAN_STD,
    ia.Reduce.NAN_VAR: ciarray.IARRAY_REDUCE_NAN_VAR,
    ia.Reduce.NAN_MEDIAN: ciarray.IARRAY_REDUCE_NAN_MEDIAN,
    ia.Reduce.ALL: ciarray.IARRAY_REDUCE_ALL,
    ia.Reduce.ANY: ciarray.IARRAY_REDUCE_ANY,
}

def reduce(cfg, a, method, axis, oneshot, correction):
    ctx = Context(cfg)

    cdef ciarray.iarray_reduce_func_t func = reduce_to_c[method]
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    iarray_check(ciarray.iarray_reduce(ctx_, a_, func, axis, &store_, &c, oneshot, correction))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    return ia.IArray(ctx, c_c)


def reduce_multi(cfg, a, method, axis, oneshot, correction):
    ctx = Context(cfg)

    cdef ciarray.iarray_reduce_func_t func = reduce_to_c[method]
    cdef ciarray.iarray_container_t *a_ = <ciarray.iarray_container_t*> PyCapsule_GetPointer(a.to_capsule(),
                                                                                             <char*>"iarray_container_t*")
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")
    cdef ciarray.iarray_container_t *c

    cdef ciarray.int8_t axis_[ciarray.IARRAY_DIMENSION_MAX]
    for i, ax in enumerate(axis):
        axis_[i] = ax

    cdef ciarray.iarray_storage_t store_
    set_storage(cfg, &store_)

    # Check that we are not inadvertently overwriting anything
    ia._check_access_mode(cfg.urlpath, cfg.mode)

    cdef ciarray.int8_t naxis = len(axis)
    iarray_check(ciarray.iarray_reduce_multi(ctx_, a_, func, naxis, axis_, &store_, &c, oneshot, correction))

    c_c = PyCapsule_New(c, <char*>"iarray_container_t*", NULL)
    cfg2 = get_cfg_from_container(cfg, ctx_, c, cfg.urlpath)
    d = ia.IArray(Context(cfg2), c_c)
    d.np_dtype = a.np_dtype

    return d


def get_ncores(max_ncores):
    cdef int ncores = 1
    try:
        iarray_check(ciarray.iarray_get_ncores(&ncores, max_ncores))
    except IArrayError:
        # In case of error, return a negative value
        return -1
    return ncores


def get_l2_size():
    cdef ciarray.uint64_t l2_size
    try:
        iarray_check(ciarray.iarray_get_L2_size(&l2_size))
    except IArrayError:
        return -1
    return l2_size


def partition_advice(dtshape, min_chunksize, max_chunksize, min_blocksize, max_blocksize, cfg):
    _dtshape = IaDTShape(dtshape).to_dict()
    cdef ciarray.iarray_dtshape_t dtshape_ = <ciarray.iarray_dtshape_t> _dtshape

    ctx = Context(cfg)
    cdef ciarray.iarray_context_t *ctx_ = <ciarray.iarray_context_t*> PyCapsule_GetPointer(ctx.to_capsule(),
                                                                                           <char*>"iarray_context_t*")

    # Create a storage struct and initialize it.  Do we really need a store for this (maybe a frame info)?
    cdef ciarray.iarray_storage_t store
    store.contiguous = False
    # Ask for the actual advice
    try:
        iarray_check(ciarray.iarray_partition_advice(ctx_, &dtshape_, &store,
                                        min_chunksize, max_chunksize, min_blocksize, max_blocksize))
    except:
        return None, None

    # Extract the shapes and return them as tuples
    chunks = tuple(store.chunkshape[i] for i in range(len(dtshape.shape)))
    blocks = tuple(store.blockshape[i] for i in range(len(dtshape.shape)))
    return chunks, blocks


# Attributes

def attr_setitem(iarr, name, content):
    cdef ciarray.iarray_container_t *c_
    c_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(iarr.to_capsule(), <char *> "iarray_container_t*")
    ctx = Context(iarr.cfg)
    cdef ciarray.iarray_context_t *ctx_
    ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(), <char *> "iarray_context_t*")

    name = name.encode("utf-8") if isinstance(name, str) else name
    cdef ciarray.bool exists
    iarray_check(ciarray.iarray_vlmeta_exists(ctx_, c_, name, &exists))

    # Fill meta
    cdef ciarray.iarray_metalayer_t meta
    a = len(content)
    if a >= 2**31:
        raise ValueError("The size of `content` cannot be larger than 2**31 - 1")
    cdef ciarray.int32_t size = a
    meta.name = name
    meta.size = size
    meta.sdata = content
    if exists:
        iarray_check(ciarray.iarray_vlmeta_update(ctx_, c_, &meta))
    else:
        iarray_check(ciarray.iarray_vlmeta_add(ctx_, c_, &meta))

def attr_getitem(iarr, name):
    cdef ciarray.iarray_container_t *c_
    c_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(iarr.to_capsule(), <char *> "iarray_container_t*")
    ctx = Context(iarr.cfg)
    cdef ciarray.iarray_context_t *ctx_
    ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(), <char *> "iarray_context_t*")

    name = name.encode("utf-8") if isinstance(name, str) else name
    cdef ciarray.bool exists = False
    iarray_check(ciarray.iarray_vlmeta_exists(ctx_, c_, name, &exists))
    if not exists:
        raise KeyError("attr does not exist")
    cdef ciarray.iarray_metalayer_t meta
    iarray_check(ciarray.iarray_vlmeta_get(ctx_, c_, name, &meta))

    return meta.sdata[:meta.size]

def attr_delitem(iarr, name):
    cdef ciarray.iarray_container_t *c_
    c_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(iarr.to_capsule(), <char *> "iarray_container_t*")
    ctx = Context(iarr.cfg)
    cdef ciarray.iarray_context_t *ctx_
    ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(), <char *> "iarray_context_t*")

    name = name.encode("utf-8") if isinstance(name, str) else name
    iarray_check(ciarray.iarray_vlmeta_delete(ctx_, c_, name))

def attr_get_names(iarr):
    cdef ciarray.iarray_container_t *c_
    c_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(iarr.to_capsule(), <char *> "iarray_container_t*")
    ctx = Context(iarr.cfg)
    cdef ciarray.iarray_context_t *ctx_
    ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(), <char *> "iarray_context_t*")
    cdef ciarray.int16_t nattrs
    iarray_check(ciarray.iarray_vlmeta_nitems(ctx_, c_, &nattrs))
    cdef char** names
    names = <char **> malloc(sizeof(char*) * nattrs)
    iarray_check(ciarray.iarray_vlmeta_get_names(ctx_, c_, names))
    res =  [names[i].decode() for i in range(nattrs)]

    free(<void*>names)
    return res

def attr_len(iarr):
    cdef ciarray.iarray_container_t *c_
    c_ = <ciarray.iarray_container_t *> PyCapsule_GetPointer(iarr.to_capsule(), <char *> "iarray_container_t*")
    ctx = Context(iarr.cfg)
    cdef ciarray.iarray_context_t *ctx_
    ctx_ = <ciarray.iarray_context_t *> PyCapsule_GetPointer(ctx.to_capsule(), <char *> "iarray_context_t*")

    cdef ciarray.int16_t nattrs
    iarray_check(ciarray.iarray_vlmeta_nitems(ctx_, c_, &nattrs))
    return nattrs


# Zarr proxy

# This function should not be called with the GIL released
cdef void zarr_handler(char *zarr_urlpath, ciarray.int64_t *slice_start, ciarray.int64_t *slice_stop,
                       ciarray.uint8_t *dest):
    path = zarr_urlpath.decode()
    z_ = _zarray_from_proxy(path)
    cdef int ndim = len(z_.shape)
    slice_ = tuple(slice(slice_start[i], slice_stop[i]) for i in range(ndim))
    data = z_[slice_]
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(data, buf, PyBUF_SIMPLE)
    memcpy(dest, buf.buf, buf.len)
    PyBuffer_Release(buf)


def set_zproxy_postfilter(iarr):
    cdef ciarray.iarray_container_t *c
    c = <ciarray.iarray_container_t *> PyCapsule_GetPointer(iarr.to_capsule(), <char *> "iarray_container_t*")
    cdef ciarray.zhandler_ptr func = zarr_handler
    urlpath = iarr.attrs["zproxy_urlpath"]
    urlpath = urlpath.encode("utf-8") if isinstance(urlpath, str) else urlpath

    iarray_check(ciarray.iarray_add_zproxy_postfilter(c, urlpath, func))


cdef class UdfLibrary:
    """
    Library for scalar UDF functions.
    """
    cdef ciarray.iarray_udf_library_t *udf_library

    def __init__(self, name):
        name = name.encode("utf-8") if isinstance(name, str) else name
        cdef ciarray.iarray_udf_library_t *library
        iarray_check(ciarray.iarray_udf_library_new(name, &library))
        self.udf_library = library

    def dealloc(self):
        ciarray.iarray_udf_library_free(&self.udf_library)

    def register_func(self, f):
        llvm_bc, dtype, arg_types, name = f.bc, f.rtype, f.argtypes, f.name
        dtype = udf2ia_dtype[dtype]
        arg_types = [udf2ia_dtype[x] for x in arg_types]
        nparr = np.array(arg_types, dtype=np.int32)
        cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
        PyObject_GetBuffer(nparr, buf, PyBUF_SIMPLE)

        name = name.encode("utf-8") if isinstance(name, str) else name
        cdef int llvm_bc_len = len(llvm_bc)
        cdef int num_args = len(arg_types)
        iarray_check(ciarray.iarray_udf_func_register(self.udf_library,
                                                      llvm_bc_len,
                                                      llvm_bc,
                                                      dtype,
                                                      num_args,
                                                      <ciarray.iarray_data_type_t *>buf.buf,
                                                      name)
                     )
        PyBuffer_Release(buf)


def udf_lookup_func(full_name):
    """Do a lookup for a `full_name` scalar UDF function.

    Parameters
    ----------
    full_name : str
        The full name of the function to be found.  Its format is ``lib_name.func_name``.

    Returns
    -------
    A pointer (int64) to the function found.  If not found, and error is raised.
    """
    full_name = full_name.encode("utf-8") if isinstance(full_name, str) else full_name
    cdef ciarray.uint64_t function_ptr
    iarray_check(ciarray.iarray_udf_func_lookup(<char *>full_name, &function_ptr))
    return function_ptr


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
    cdef int n = len(x)
    poly_nogil(&x[0], &y[0], n)
    return y.reshape(shape)

# TODO: End of the benchmarking code
