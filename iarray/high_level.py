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

import numpy as np
import numexpr as ne
import iarray as ia
from iarray import iarray_ext as ext
from itertools import zip_longest as zip


def fuse_operands(operands1, operands2):
    new_operands = {}
    dup_operands = {}
    for k2, v2 in operands2.items():
        try:
            k1 = list(operands1.keys())[list(operands1.values()).index(v2)]
            # The operand is duplicated; keep track of it
            dup_operands[k2] = k1
        except ValueError:
            # The value is not among operands1, so rebase it
            prev_pos = int(k2[1:])
            new_pos = prev_pos + len(new_operands)
            new_op = f"o{new_pos}"
            new_operands[new_op] = operands2[k2]
    return new_operands, dup_operands


def fuse_expressions(expr, new_base, dup_op):
    new_expr = ""
    skip_to_char = 0
    for i in range(len(expr)):
        if i < skip_to_char:
            continue
        if expr[i] == 'o':
            try:
                j = expr[i+1:].index(' ')
            except ValueError:
                j = expr[i + 1:].index(')')
            old_pos = int(expr[i+1:i+j+1])
            old_op = f"o{old_pos}"
            if old_op not in dup_op:
                new_pos = old_pos + new_base
                new_expr += f"o{new_pos}"
            else:
                new_expr += dup_op[old_op]
            skip_to_char = i + j + 1
        else:
            new_expr += expr[i]
    return new_expr


class RandomContext(ext.RandomContext):

    def __init__(self, **kwargs):
        cfg = Config(**kwargs)
        super(RandomContext, self).__init__(cfg)


class Config(ext._Config):

    def __init__(self, compression_codec=ia.LZ4, compression_level=5, use_dict=0, filter_flags=ia.SHUFFLE,
                 max_num_threads=1, fp_mantissa_bits=0, blocksize=0, eval_flags="iterblock"):
        self._compression_codec = compression_codec
        self._compression_level = compression_level
        self._use_dict = use_dict
        self._filter_flags = filter_flags
        self._eval_flags = eval_flags
        self._max_num_threads = max_num_threads
        self._fp_mantissa_bits = fp_mantissa_bits
        self._blocksize = blocksize
        self._eval_flags = eval_flags  # TODO: should we move this to its own eval configuration?
        super(Config, self).__init__(compression_codec, compression_level, use_dict, filter_flags,
                                     max_num_threads, fp_mantissa_bits, blocksize, eval_flags)

    @property
    def compression_codec(self):
        codec = ["BloscLZ", "LZ4", "LZ4HC", "Snappy", "Zlib", "Zstd", "Lizard"]
        return codec[self._compression_codec]

    @property
    def compression_level(self):
        return self._compression_level

    @property
    def filter_flags(self):
        flags = {0: "NOFILTER", 1: "SHUFFLE", 2: "BITSHUFFLE", 4: "DELTA", 8: "TRUNC_PREC"}
        return flags[self._filter_flags]

    @property
    def max_num_threads(self):
        return self._max_num_threads

    @property
    def fp_mantissa_bits(self):
        return self._fp_mantissa_bits

    @property
    def blocksize(self):
        return self._blocksize

    @property
    def eval_flags(self):
        return self._eval_flags

    def __str__(self):
        res = f"IArray Config object:\n"
        compression_codec = f"    Compression codec: {self.compression_codec}\n"
        compression_level = f"    Compression level: {self.compression_level}\n"
        filter_flags = f"    Filter flags: {self.filter_flags}\n"
        eval_flags = f"    Eval flags: {self.eval_flags}\n"
        max_num_threads = f"    Max. num. threads: {self.max_num_threads}\n"
        fp_mantissa_bits = f"    Fp mantissa bits: {self.fp_mantissa_bits}\n"
        blocksize = f"    Blocksize: {self.blocksize}"
        return res + compression_codec + compression_level + filter_flags + eval_flags + \
               max_num_threads + fp_mantissa_bits + blocksize


class LazyExpr:

    def __init__(self, new_op):
        # This is the very first time that a LazyExpr is formed from two operands that are not LazyExpr themselves
        value1, op, value2 = new_op
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            self.expression = f"({value1} {op} {value2})"
        elif isinstance(value2, (int, float)):
            self.operands = {"o0": value1}
            self.expression = f"(o0 {op} {value2})"
        elif isinstance(value1, (int, float)):
            self.operands = {"o0": value2}
            self.expression = f"({value1} {op} o0)"
        else:
            if value1 == value2:
                self.operands = {"o0": value1}
                self.expression = f"(o0 {op} o0)"
            elif isinstance(value1, LazyExpr) or isinstance(value2, LazyExpr):
                if isinstance(value1, LazyExpr):
                    self.expression = value1.expression
                    self.operands = {"o0": value2}
                else:
                    self.expression = value2.expression
                    self.operands = {"o0": value1}
                self.update_expr(new_op)
            else:
                self.operands = {"o0": value1, "o1": value2}
                self.expression = f"(o0 {op} o1)"

    def update_expr(self, new_op):
        # One of the two operands are LazyExpr instances
        value1, op, value2 = new_op
        if isinstance(value1, LazyExpr) and isinstance(value2, LazyExpr):
            # Expression fusion
            # Fuse operands in expressions and detect duplicates
            new_op, dup_op = fuse_operands(value1.operands, value2.operands)
            self.operands.update(new_op)
            # Take expression 2 and rebase the operands while removing duplicates
            new_expr = fuse_expressions(value2.expression, len(value1.operands), dup_op)
            self.expression = f"({self.expression} {op} {new_expr})"
        elif isinstance(value1, LazyExpr):
            if isinstance(value2, (int, float)):
                self.expression = f"({self.expression} {op} {value2})"
            else:
                try:
                    op_name = list(value1.operands.keys())[list(value1.operands.values()).index(value2)]
                except ValueError:
                    op_name = f"o{len(self.operands)}"
                    self.operands[op_name] = value2
                self.expression = f"({self.expression} {op} {op_name})"
        else:
            if isinstance(value1, (int, float)):
                self.expression = f"({value1} {op} {self.expression})"
            else:
                try:
                    op_name = list(value2.operands.keys())[list(value2.operands.values()).index(value1)]
                except ValueError:
                    op_name = f"o{len(self.operands)}"
                    self.operands[op_name] = value1
                self.expression = f"({op_name} {op} {self.expression})"
        return self


    def __add__(self, value):
        return self.update_expr(new_op=(self, '+', value))

    def __radd__(self, value):
        return self.update_expr(new_op=(value, '+', self))

    def __sub__(self, value):
        return self.update_expr(new_op=(self, '-', value))

    def __rsub__(self, value):
        return self.update_expr(new_op=(value, '-', self))

    def __mul__(self, value):
        return self.update_expr(new_op=(self, '*', value))

    def __rmul__(self, value):
        return self.update_expr(new_op=(value, '*', self))

    def __truediv__(self, value):
        return self.update_expr(new_op=(self, '/', value))

    def __rtruediv__(self, value):
        return self.update_expr(new_op=(value, '/', self))


    def eval(self, method="iarray_eval", **kwargs):
        # TODO: see if shape and pshape can be instance variables, or better stay like this
        o0 = self.operands['o0']
        shape_ = o0.shape
        pshape_ = o0.pshape
        if method == "iarray_eval":
            expr = Expr(**kwargs)
            for k, v in self.operands.items():
                if isinstance(v, IArray):
                    expr.bind(k, v)
            expr.compile(self.expression)
            out = expr.eval(shape_, pshape_, "double")
        elif method == "numexpr":
            out = ia.empty(shape=shape_, pshape=pshape_, **kwargs)
            operand_iters = tuple(o.iter_read_block(pshape_) for o in self.operands.values() if isinstance(o, IArray))
            all_iters = operand_iters + (out.iter_write_block(pshape_),)   # put the iterator for the output at the end
            # all_iters =  (out.iter_write_block(pshape_),) + operand_iters  # put the iterator for the output at the front
            for block in zip(*all_iters):
                block_operands = {o: block[i][1] for (i, o) in enumerate(self.operands.keys(), start=0)}
                out_block = block[-1][1]  # the block for output is at the end, by construction
                # block_operands = {o: block[i][1] for (i, o) in enumerate(self.operands.keys(), start=1)}
                # out_block = block[0][1]  # the block for output is at the front, by construction
                ne.evaluate(self.expression, local_dict=block_operands, out=out_block)
        else:
            raise ValueError(f"Unrecognized '{method}' method")

        return out


    def __str__(self):
        expression = f"{self.expression}"
        return expression


# The main IronArray container (not meant to be called from user space)
class IArray(ext.Container):

    def __add__(self, value):
        return LazyExpr(new_op=(self, '+', value))

    def __radd__(self, value):
        return LazyExpr(new_op=(value, '+', self))

    def __sub__(self, value):
        return LazyExpr(new_op=(self, '-', value))

    def __rsub__(self, value):
        return LazyExpr(new_op=(value, '-', self))

    def __mul__(self, value):
        return LazyExpr(new_op=(self, '*', value))

    def __rmul__(self, value):
        return LazyExpr(new_op=(value, '*', self))

    def __truediv__(self, value):
        return LazyExpr(new_op=(self, '/', value))

    def __rtruediv__(self, value):
        return LazyExpr(new_op=(value, '/', self))


# The main expression class
class Expr(ext.Expression):

    def __init__(self, **kwargs):
        cfg = Config(**kwargs)
        super(Expr, self).__init__(cfg)


#
# Constructors
#

def empty(shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.empty(cfg, shape, pshape, dtype, filename)


def arange(start=None, stop=None, step=None, shape=None, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)

    if stop is None and step is None:
        stop = start
        start = 0
        step = 1
    elif step is None:
        step = 1

    slice_ = slice(start, stop, step)
    return ext.arange(cfg, slice_, shape, pshape, dtype, filename)


def linspace(nelem, start, stop, shape=None, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.linspace(cfg, nelem, start, stop, shape, pshape, dtype, filename)


def zeros(shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.zeros(cfg, shape, pshape, dtype, filename)


def ones(shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.ones(cfg, shape, pshape, dtype, filename)


def full(fill_value, shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.full(cfg, fill_value, shape, pshape, dtype, filename)


def from_file(filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.from_file(cfg, filename)


def iarray2numpy(c, **kwargs):
    cfg = Config(**kwargs)
    return ext.iarray2numpy(cfg, c)


def numpy2iarray(c, pshape=None, filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.numpy2iarray(cfg, c, pshape, filename)


def random_rand(shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_rand(cfg, shape, pshape, dtype, filename)


def random_randn(shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_randn(cfg, shape, pshape, dtype, filename)


def random_beta(alpha, beta, shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_beta(cfg, alpha, beta, shape, pshape, dtype, filename)


def random_lognormal(mu, sigma, shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_lognormal(cfg, mu, sigma, shape, pshape, dtype, filename)


def random_exponential(beta, shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_exponential(cfg, beta, shape, pshape, dtype, filename)


def random_uniform(a, b, shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_uniform(cfg, a, b, shape, pshape, dtype, filename)


def random_normal(mu, sigma, shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_normal(cfg, mu, sigma, shape, pshape, dtype, filename)


def random_bernoulli(p, shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_bernoulli(cfg, p, shape, pshape, dtype, filename)


def random_binomial(m, p, shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_binomial(cfg, m, p, shape, pshape, dtype, filename)


def random_poisson(l, shape, pshape=None, dtype="double", filename=None, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_poisson(cfg, l, shape, pshape, dtype, filename)


def random_kstest(a, b, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_kstest(cfg, a, b)


def matmul(a, b, block_a, block_b, **kwargs):
    cfg = Config(**kwargs)
    return ext.matmul(cfg, a, b, block_a, block_b)


if __name__ == "__main__":
    # Define array params
    _shape = [40]
    _pshape = [20]
    size = int(np.prod(_shape))

    # Create initial containers
    a1 = ia.linspace(size, 0, 10, _shape, _pshape, "double")
    a2 = ia.linspace(size, 0, 20, _shape, _pshape, "double")
    # a3 = a1 + a2 + a1 - 2 * a1 + 1
    a3 = a1 + 2 * a1 + 1
    # a3 = a1 + a2
    print(a3)
    a3 += 2
    print(a3)
    # a4 = a3.eval(method="numexpr")
    a4 = a3.eval(method="iarray_eval")
    a4_np = ia.iarray2numpy(a4)
    print(a4_np)
