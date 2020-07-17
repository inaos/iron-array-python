# Standard Library
import math

# Requirements
import iarray as ia
from llvmlite import ir
from . import py2llvm
from .py2llvm import int8, int8p, int32, int64, int32p, int64p
from .py2llvm import types

from . import iarray_ext

assert math  # Silence pyflakes warning


# From iarray/iarray-c-develop/src/iarray_expression.c
IARRAY_EXPR_OPERANDS_MAX = 128
class udf_type(types.StructType):
    _name_ = 'iarray_eval_pparams_t'
    _fields_ = [
        ('ninputs', int32),  # int32 may not be the same as int
        ('inputs', ir.ArrayType(int8p, IARRAY_EXPR_OPERANDS_MAX)),
        ('input_typesizes', ir.ArrayType(int32, IARRAY_EXPR_OPERANDS_MAX)),
        ('user_data', int8p),  # LLVM does not have the concept of void*
        ('out', int8p),  # LLVM doesn't make the difference between signed and unsigned
        ('out_size', int32),
        ('out_typesize', int32),  # int32_t out_typesize;  // automatically filled
        ('ndim', int8),
        ('window_shape', int32p),
        ('window_start', int64p),
        ('window_strides', int32p),
    ]


class ArrayShape(types.ArrayShape):

    def __init__(self, name, shape, array):
        self.name = name
        self.shape = shape
        self.array = array

    def get(self, visitor, n):
        builder = visitor.builder
        n_ir = types.value_to_ir_value(builder, n, type_=int8)

        # Check bounds
        ndim = self.array.function._ndim
        test = builder.icmp_signed('>=', n_ir, ndim)
        with builder.if_then(test, likely=False):
            return_type = builder.function.type.pointee.return_type
            error = ir.Constant(return_type, iarray_ext.IARRAY_ERR_EVAL_ENGINE_OUT_OF_RANGE)
            builder.ret(error)

        # General case
        name = f'{self.name}_{n}'
        size = builder.gep(self.shape, [n_ir])  # i64*
        size = builder.load(size, name=name)  # i64
        return size


class ArrayType(types.ArrayType):

    def __init__(self, function, name, args):
        self.function = function
        self.name = name
        self.window_shape = ArrayShape('window_shape', self._shape, self)
        self.window_start = ArrayShape('window_start', self._start, self)
        self.window_strides = ArrayShape('window_strides', self._strides, self)

    def preamble(self, builder):
        if self.idx == 0:
            # .out (uint8_t*)
            ptr = self.function._out
        else:
            # .inputs (uint8_t**)
            ptr = self.function.get_field(builder, 1, name='inputs')
            # .inputs[n] (uint8_t*)
            idx = ir.Constant(int32, self.idx - 1)
            ptr = builder.gep(ptr, [types.zero, idx])
            ptr = builder.load(ptr)

        # Cast
        self.ptr = builder.bitcast(ptr, self.dtype.as_pointer(), name=self.name)

    @property
    def _shape(self):
        return self.function._shape

    @property
    def _start(self):
        return self.function._start

    @property
    def _strides(self):
        return self.function._strides

    def get_ptr(self, visitor):
        return self.ptr

    # For compatibility with numpy arrays
    @property
    def shape(self):
        return self.window_shape

    @property
    def strides(self):
        return self.window_strides


def Array(dtype, ndim):
    return type(
        f'Array[{dtype}, {ndim}]',
        (ArrayType,),
        dict(dtype=dtype, ndim=ndim)
    )


class Function(py2llvm.Function):

    def get_py_signature(self, signature):
        signature = super().get_py_signature(signature)
        for i, param in enumerate(signature.parameters):
            param.type.idx = i
        return signature

    def get_ir_signature(self, node, verbose=0, *args):
        dtype = self.llvm.get_dtype(self.ir_module, udf_type)
        params = [py2llvm.Parameter('params', dtype)]

        return_type = types.type_to_ir_type(int64)
        return py2llvm.Signature(params, return_type)

    def preamble(self, builder, args):
        params = args['params']
        self.params_ptr = builder.load(params)  # iarray_eval_pparams_t*
        # self._ninputs = self.load_field(builder, 0, name='ninputs')
        # self._inputs = self.load_field(builder, 1, name='ninputs')                  # i8**
        # self._input_typesizes = self.load_field(builder, 2, name='input_typesizes') # i8*
        # self._user_data = self.load_field(builder, 3, name='user_data')             # i8*
        self._out = self.load_field(builder, 4, name='out')                         # i8*
        # self._out_size = self.load_field(builder, 5, name='out_size')               # i32
        # self._out_typesize = self.load_field(builder, 6, name='out_typesize')       # i32
        self._ndim = self.load_field(builder, 7, name='ndim')                       # i8
        self._shape = self.load_field(builder, 8, name='window_shape')              # i32*
        self._start = self.load_field(builder, 9, name='window_start')              # i64*
        self._strides = self.load_field(builder, 10, name='window_strides')         # i32*

    def get_field(self, builder, idx, name=''):
        idx = ir.Constant(int32, idx)
        return builder.gep(self.params_ptr, [types.zero32, idx], name=name)

    def load_field(self, builder, idx, name=''):
        ptr = self.get_field(builder, idx)
        return builder.load(ptr, name=name)

    def create_expr(self, inputs, dtshape, method='auto', **cparams):
        eval_flags = ia.EvalFlags(method=method, engine="compiler")
        expr = ia.Expr(eval_flags=eval_flags, **cparams)
        for a in inputs:
            expr.bind("", a)
        cfg = ia.Config(**cparams)
        expr.bind_out_properties(dtshape, cfg._storage)
        expr.compile_udf(self)
        return expr


class LLVM(py2llvm.LLVM):

    def jit(self, *args, **kwargs):
        kwargs['optimize'] = False  # iron-array optimizes, not py2llvm
        return super().jit(*args, **kwargs)


llvm = LLVM(Function)
jit = llvm.jit
