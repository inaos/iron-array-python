# Standard Library
import argparse
import math

# Requirements
import iarray as ia
from llvmlite import ir
import py2llvm
from py2llvm import int8, int8p, int32, int64, int64p
from py2llvm import types


assert math # Silence pyflakes warning

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='count', default=0)
args = parser.parse_args()
verbose = args.verbose


"""
typedef struct iarray_eval_pparams_s {
    int ninputs;  // number of data inputs
    uint8_t* inputs[IARRAY_EXPR_OPERANDS_MAX];  // the data inputs
    int32_t input_typesizes[IARRAY_EXPR_OPERANDS_MAX];  // the typesizes for data inputs
    void *user_data;  // a pointer to an iarray_expr_pparams_t struct
    uint8_t *out;  // the output buffer
    int32_t out_size;  // the size of output buffer (in bytes)
    int32_t out_typesize;  // the typesize of output
    int8_t ndim;  // the number of dimensions for inputs / output arrays
    int64_t *vis_shape;  // the visible shape of the input arrays (NULL if not available)
    int64_t *elem_index; // the starting index for the visible shape (NULL
} iarray_eval_pparams_t;

/**
 * @brief The type of the prefilter function.
 *
 * If the function call is successful, the return value should be 0; else, a negative value.
 */
#typedef int (*blosc2_prefilter_fn)(iarray_eval_pparams_t* params);
"""

BLOSC2_PREFILTER_INPUTS_MAX = 128
class udf_type(types.StructType):
    _name_ = 'iarray_eval_pparams_t'
    _fields_ = [
        ('ninputs', int32), # int32 may not be the same as int
        ('inputs', ir.ArrayType(int8p, BLOSC2_PREFILTER_INPUTS_MAX)),
        ('input_typesizes', ir.ArrayType(int32, BLOSC2_PREFILTER_INPUTS_MAX)),
        ('user_data', int8p), # LLVM does not have the concept of void*
        ('out', int8p), # LLVM doesn't make the difference between signed and unsigned
        ('out_size', int32),
        ('out_typesize', int32), # int32_t out_typesize;  // automatically filled
        ('ndim', int8),
        ('vis_shape', int64p),
        ('elem_index', int64p),
    ]


class ArrayShape(types.ArrayShape):

    def __init__(self, array):
        self.array = array

    def get(self, visitor, n):
        builder = visitor.builder

        # XXX Old code, when we didn't have access to the pshape
        if self.array.idx == 0:
            import ast
            out_size = self.array.get_field(builder, 5)                    # i32*
            out_size = builder.load(out_size, name='out_size')             # i32
            out_typesize = self.array.get_field(builder, 6)                # i32*
            out_typesize = builder.load(out_typesize, name='out_typesize') # i32
            return visitor.BinOp_exit(None, None, out_size, ast.Div, out_typesize)

        # All arrays, input and output have the same phsape
        n = ir.Constant(int32, n)
        size = builder.gep(self.array.pshape, [n]) # i64*
        size = builder.load(size)                  # i64
        return size

        # We don't use this yet, anywhere, ndim is got from the type hint
#       ndim = self.array.get_field(builder, 7) # i8*
#       ndim = builder.load(ndim, name='ndim')  # i8


class ArrayType(types.ArrayType):

    def __init__(self, name, args):
        self.name = name
        self.params = args['params'] # iarray_eval_pparams_t**
        self.shape = ArrayShape(self)

    def preamble(self, builder):
        self.params_ptr = builder.load(self.params)      # iarray_eval_pparams_t*
        pshape = self.get_field(builder, 8)              # i64**
        self.pshape = builder.load(pshape, name='shape') # i64*
        if self.idx == 0:
            # .out (uint8_t*)
            ptr = self.get_field(builder, 4)
            ptr = builder.load(ptr)
        else:
            # .inputs (uint8_t**)
            ptr = self.get_field(builder, 1, name='inputs')
            # .inputs[n] (uint8_t*)
            idx = ir.Constant(int32, self.idx - 1)
            ptr = builder.gep(ptr, [types.zero, idx])
            ptr = builder.load(ptr)

        # Cast
        self.ptr = builder.bitcast(ptr, self.dtype.as_pointer(), name=self.name)

    def get_field(self, builder, idx, name=''):
        idx = ir.Constant(int32, idx)
        return builder.gep(self.params_ptr, [types.zero32, idx], name=name)

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
