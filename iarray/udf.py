# Standard Library
import argparse
import ast

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
        out_size = self.array.get_field(visitor, 5) # gep
        out_typesize = self.array.get_field(visitor, 6) # load
        out_size = visitor.builder.load(out_size) # gep
        out_typesize = visitor.builder.load(out_typesize) # load
        return visitor.BinOp_exit(None, None, out_size, ast.Div, out_typesize)


class ArrayType(types.ArrayType):

    def __init__(self, name, args):
        self.name = name
        self.params = args['params']
        self.shape = ArrayShape(self)

    def get_field(self, visitor, idx):
        idx = ir.Constant(int32, idx)
        ptr = visitor.builder.load(self.params)
        ptr = visitor.builder.gep(ptr, [types.zero32, idx])
        return ptr

    def get_ptr(self, visitor):
        if self.idx == 0:
            # .out (uint8_t*)
            ptr = self.get_field(visitor, 4)
            ptr = visitor.builder.load(ptr)
        else:
            # .inputs (uint8_t**)
            ptr = self.get_field(visitor, 1)
            # .inputs[n] (uint8_t*)
            idx = ir.Constant(int32, self.idx - 1)
            ptr = visitor.builder.gep(ptr, [types.zero, idx])
            ptr = visitor.builder.load(ptr)

        # Cast
        return visitor.builder.bitcast(ptr, self.dtype.as_pointer())


def Array(dtype, ndim):
    return type(
        f'Array[{dtype}, {ndim}]',
        (ArrayType,),
        dict(dtype=dtype, ndim=ndim)
    )


class UDFFunction(Function):

    def get_ir_signature(self, node, verbose=0, *args):
        dtype = self.llvm.get_dtype(self.ir_module, udf_type)
        params = [Parameter('params', dtype)]

        return_type = types.type_to_ir_type(types.int64)
        return Signature(params, return_type)

    def get_py_signature(self, signature):
        signature = super().get_py_signature(signature)
        for i, param in enumerate(signature.parameters):
            param.type.idx = i
        return signature

    def create_expr(self, inputs, **cparams):
        expr = ia.Expr(eval_flags="iterblosc", **cparams)
        for a in inputs:
            expr.bind("", a)

        expr.compile_udf(self)
        return expr


llvm = LLVM(UDFFunction)
jit = llvm.jit
