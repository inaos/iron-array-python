# Standard Library
import argparse
import ast
import math

# Requirements
import iarray as ia
from llvmlite import ir
import py2llvm
from py2llvm import int8p, int32, int64
from py2llvm import types


assert math # Silence pyflakes warning

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


class ArrayShape(types.ArrayShape):

    def __init__(self, array):
        self.array = array

    def get(self, visitor, n):
        # The dimension size is the same for every dimension in every array
        builder = visitor.builder
        out_size = self.array.get_field(builder, 5) # gep
        out_typesize = self.array.get_field(builder, 6) # load
        out_size = builder.load(out_size) # gep
        out_typesize = builder.load(out_typesize) # load
        return visitor.BinOp_exit(None, None, out_size, ast.Div, out_typesize)


class ArrayType(types.ArrayType):

    def __init__(self, name, args):
        self.name = name
        self.params = args['params']
        self.shape = ArrayShape(self)

    def preamble(self, builder):
        if self.idx == 0:
            # .out (uint8_t*)
            ptr = self.get_field(builder, 4)
            ptr = builder.load(ptr)
        else:
            # .inputs (uint8_t**)
            ptr = self.get_field(builder, 1)
            # .inputs[n] (uint8_t*)
            idx = ir.Constant(int32, self.idx - 1)
            ptr = builder.gep(ptr, [types.zero, idx])
            ptr = builder.load(ptr)

        # Cast
        self.ptr = builder.bitcast(ptr, self.dtype.as_pointer())

    def get_field(self, builder, idx):
        idx = ir.Constant(int32, idx)
        ptr = builder.load(self.params)
        return builder.gep(ptr, [types.zero32, idx])

    def get_ptr(self, visitor):
        return self.ptr


def Array(dtype, ndim):
    return type(
        f'Array[{dtype}, {ndim}]',
        (ArrayType,),
        dict(dtype=dtype, ndim=ndim)
    )


class Function(py2llvm.Function):

    def get_ir_signature(self, node, verbose=0, *args):
        dtype = self.llvm.get_dtype(self.ir_module, udf_type)
        params = [py2llvm.Parameter('params', dtype)]

        return_type = types.type_to_ir_type(int64)
        return py2llvm.Signature(params, return_type)

    def get_py_signature(self, signature):
        signature = super().get_py_signature(signature)
        for i, param in enumerate(signature.parameters):
            param.type.idx = i
        return signature

    def create_expr(self, inputs, dtshape, **cparams):
        eval_flags = ia.EvalFlags(method="iterblosc", engine="compiler")
        expr = ia.Expr(eval_flags=eval_flags, **cparams)
        for a in inputs:
            expr.bind("", a)
        cfg = ia.Config(**cparams)
        expr.bind_out_properties(dtshape, cfg._storage)
        expr.compile_udf(self)
        return expr


class LLVM(py2llvm.LLVM):

    def jit(self, *args, **kwargs):
        kwargs['optimize'] = False # iron-array optimizes, not py2llvm
        return super().jit(*args, **kwargs)


llvm = LLVM(Function)
jit = llvm.jit
