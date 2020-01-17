# Standard Library
import argparse
import ast
import ctypes
from ctypes import c_int, c_uint8, c_int32
import inspect

# Requirements
import iarray as ia
from llvmlite import ir
from py2llvm import int8p, int32
from py2llvm import types
from py2llvm import LLVM, Function, Signature, Parameter


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='count', default=0)
args = parser.parse_args()
verbose = args.verbose


"""
typedef struct {
  int ninputs;  // number of data inputs
  uint8_t* inputs[BLOSC2_PREFILTER_INPUTS_MAX];  // the data inputs
  int32_t input_typesizes[BLOSC2_PREFILTER_INPUTS_MAX];  // the typesizes for data inputs
  void *user_data;  // user-provided info (optional)
  uint8_t *out;  // automatically filled
  int32_t out_size;  // automatically filled
  int32_t out_typesize;  // automatically filled
} blosc2_prefilter_params;

/**
 * @brief The type of the prefilter function.
 *
 * If the function call is successful, the return value should be 0; else, a negative value.
 */
#typedef int (*blosc2_prefilter_fn)(blosc2_prefilter_params* params);
"""

BLOSC2_PREFILTER_INPUTS_MAX = 128
c_uint8_p = ctypes.POINTER(c_uint8)
inputs_type = c_uint8_p * BLOSC2_PREFILTER_INPUTS_MAX
input_typesizes_type = c_int32 * BLOSC2_PREFILTER_INPUTS_MAX
class blosc2_prefilter_params(ctypes.Structure):
    _fields_ = [
        ('ninputs', c_int),
        ('inputs', inputs_type),
        ('input_typesizes', input_typesizes_type),
        ('user_data', ctypes.c_void_p),
        ('out', c_uint8_p),
        ('out_size', c_int32),
        ('out_typesize', c_int32),
    ]


class udf_array_shape:

    def __init__(self, params):
        self.params = params

    def subscript(self, visitor, slice, ctx):
        assert ctx is ast.Load
#       assert slice == 0 # XXX For now we only support 1 dimension arrays

        # The dimension size is the same for every dimension in every array
        params = self.params
        out_size = params.out_size(visitor) # gep
        out_typesize = params.out_typesize(visitor) # load
        out_size = visitor.builder.load(out_size) # gep
        out_typesize = visitor.builder.load(out_typesize) # load
        return visitor.BinOp_exit(None, None, out_size, ast.Div, out_typesize)

class udf_array:

    def __init__(self, params, idx, dtype, ndim):
        self.params = params
        self.idx = idx
        self.dtype = dtype
        self.ndim = ndim

    @property
    def shape(self):
        return udf_array_shape(self.params)

    def subscript(self, visitor, slice, ctx):
        params = self.params
        if self.idx is None:
            assert ctx is ast.Store
            arr = params.get_out(visitor, slice)
        else:
            assert ctx is ast.Load
            arr = params.get_input(visitor, self.idx, slice)

        return arr


class udf_type(types.StructType):
    _name_ = 'blosc2_prefilter_params'
    _fields_ = [
        ('ninputs', int32), # int32 may not be the same as int
        ('inputs', ir.ArrayType(int8p, BLOSC2_PREFILTER_INPUTS_MAX)),
        ('input_typesizes', ir.ArrayType(int32, BLOSC2_PREFILTER_INPUTS_MAX)),
        ('user_data', int8p), # LLVM does not have the concept of void*
        ('out', int8p), # LLVM doesn't make the difference between signed and unsigned
        ('out_size', int32),
        ('out_typesize', int32), # int32_t out_typesize;  // automatically filled
    ]

    def get_locals(self):
        return {
            name: udf_array(self, idx, dtype, ndim)
            for name, idx, dtype, ndim in self.local_vars
        }

    def get_input(self, visitor, n, idx):
        # .inputs (uint8_t**)
        inputs = super().__getattr__('inputs')
        ptr = inputs(visitor)
        # .inputs[n] (uint8_t*)
        n_ir = types.value_to_ir_value(n)
        ptr = visitor.builder.gep(ptr, [types.zero, n_ir])
        ptr = visitor.builder.load(ptr)
        # .inputs[n] (<type>*)
        typ = self.input_types[n]
        ptr = visitor.builder.bitcast(ptr, typ.as_pointer())
        # .inputs[n][slice] (<type>)
        ptr = visitor.builder.gep(ptr, [idx])
        return visitor.builder.load(ptr)

    def get_out(self, visitor, idx):
        # .out (uint8_t*)
        out = super().__getattr__('out')
        ptr = out(visitor)
        # .out (<type>*)
        ptr = visitor.builder.load(ptr)
        typ = self.out_type
        ptr = visitor.builder.bitcast(ptr, typ.as_pointer())
        # .out[n]
        return visitor.builder.gep(ptr, [idx])


def udf_type_factory(local_vars):
    out_type = local_vars[0][2]
    input_types = [x[2] for x in local_vars[1:]]

    return type(
        f'udf_type[]',
        (udf_type,),
        dict(
            local_vars=local_vars,
            input_types=input_types,
            out_type=out_type,
        )
    )


class UDFFunction(Function):

    def _get_signature(self, signature):
        assert signature is None
        signature = inspect.signature(self.py_function)
        parameters = list(signature.parameters.values())
        assert len(parameters) >= 2

        local_vars = []
        for idx, param in enumerate(parameters):
            name = param.name
            annotation = param.annotation
            assert param.default == inspect.Parameter.empty
            assert issubclass(annotation, types.ArrayType)
            idx = (idx - 1) if idx > 0 else None
            local_vars.append((name, idx, annotation.dtype, annotation.ndim))

        parameters = [Parameter('params', udf_type_factory(local_vars))]
        return Signature(parameters, types.int64)

    def create_expr(self, inputs, **cparams):
        expr = ia.Expr(eval_flags="iterblosc", **cparams)
        for a in inputs:
            expr.bind("", a)

        expr.compile_udf(self)
        return expr


llvm = LLVM(UDFFunction)
jit = llvm.jit
