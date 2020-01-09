import argparse
import ctypes
from ctypes import c_int, c_uint8, c_int32

from llvmlite import ir

#import py2llvm as llvm
from py2llvm import StructType, int8p, int32, float64
from py2llvm import types


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


class params_type_input:

    def __init__(self, ptr, typ):
        self.ptr = ptr
        self.typ = typ

    def to_ir_value(self, visitor):
        ptr = visitor.builder.load(self.ptr)
        ptr = visitor.builder.bitcast(ptr, self.typ.as_pointer())
        return ptr

    def subscript(self, visitor, slice, ctx):
        ptr = self.to_ir_value(visitor)
        ptr = visitor.builder.gep(ptr, [slice])
        return visitor.builder.load(ptr)


class params_type_inputs:

    def __init__(self, ptr, typs):
        self.ptr = ptr
        self.typs = typs

    def subscript(self, visitor, slice, ctx):
        typ = self.typs[slice]
        idx = types.value_to_ir_value(slice)
        ptr = visitor.builder.gep(self.ptr, [types.zero, idx])
        return params_type_input(ptr, typ)

class params_type_out:

    def __init__(self, ptr, typ):
        self.ptr = ptr
        self.typ = typ

    def subscript(self, visitor, slice, ctx):
        ptr = visitor.builder.load(self.ptr)
        ptr = visitor.builder.bitcast(ptr, self.typ.as_pointer())
        ptr = visitor.builder.gep(ptr, [slice])
        return ptr

class params_type(StructType):
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

    input_types = [float64]
    out_type = float64

    @property
    def inputs(self):
        inputs = super().__getattr__('inputs')
        def cb(visitor):
            ptr = inputs(visitor)
            return params_type_inputs(ptr, self.input_types)

        return cb

    @property
    def out(self):
        out = super().__getattr__('out')
        def cb(visitor):
            ptr = out(visitor)
            return params_type_out(ptr, self.out_type)

        return cb
