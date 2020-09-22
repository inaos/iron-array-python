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
from collections import namedtuple
import warnings


def get_ncores(max_ncores=0):
    """Return the number of logical cores in the system.

    This number is capped at `max_ncores`.  When `max_ncores` is 0,
    there is no cap at all.
    """
    ncores = ext.get_ncores(max_ncores)
    if ncores < 0:
        warnings.warn("Error getting the number of cores in this system (please report this)."
                      "  Falling back to 1.",
                      UserWarning)
        return 1
    return ncores


def partition_advice(dtshape, min_chunksize=0, max_chunksize=0, min_blocksize=0, max_blocksize=0):
    """
    Provide advice for the chunk and block shapes for a certain `dtshape`.

    If success, the tuple (chunkshape, blockshape) containing the advice is returned.

    `min_` and `max_` params contain minimum and maximum values for chunksize and blocksize.
    If `min_` or `max_` are 0, they default to sensible values (fractions of CPU caches).

    In case of error, a (None, None) tuple is returned and a warning is issued.
    """
    chunkshape, blockshape = ext.partition_advice(dtshape, min_chunksize, max_chunksize,
                                                  min_blocksize, max_blocksize)
    if chunkshape is None:
        warnings.warn("Error in providing partition advice (please report this)."
                      "  Please do not trust on the chunkshape and blockshape in `storage`!",
                      UserWarning)
    return chunkshape, blockshape


def cmp_arrays(a, b, success=None):
    if type(a) is ia.high_level.IArray:
        a = ia.iarray2numpy(a)

    if type(b) is ia.high_level.IArray:
        b = ia.iarray2numpy(b)

    if a.dtype == np.float64 and b.dtype == np.float64:
        tol = 1e-14
    else:
        tol = 1e-6
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)

    if success is not None:
        print(success)


# TODO: add docstrings
class dtshape:

    def __init__(self, shape=None, dtype=np.float64):
        if shape is None:
            return ValueError("shape cannot be None")
        dtype = np.dtype(dtype)
        if dtype.type not in (np.float32, np.float64):
            raise NotImplementedError("Only float32 and float64 types are supported for now")
        self.shape = shape
        self.dtype = dtype

    def to_tuple(self):
        Dtshape = namedtuple('dtshape', 'shape dtype')
        return Dtshape(self.shape, self.dtype)


# TODO: add docstrings
class Config(ext._Config):

    def __init__(self, clib=ia.LZ4, clevel=5, use_dict=0, filter_flags=ia.SHUFFLE, nthreads=0,
                 fp_mantissa_bits=0, storage=None, eval_method=None, seed=0):
        self._clib = clib
        self._clevel = clevel
        self._use_dict = use_dict
        self._fp_mantissa_bits = fp_mantissa_bits
        if fp_mantissa_bits > 0:
            filter_flags |= ia.TRUNC_PREC
        self._filter_flags = filter_flags
        # Get the number of cores using nthreads as a maximum
        self._nthreads = nthreads = get_ncores(nthreads)
        self._seed = seed
        # TODO: should we move this to its own eval configuration?
        self._eval_method = ia.EVAL_AUTO if eval_method is None else eval_method
        self._storage = ia.StorageProperties() if storage is None else storage
        super().__init__(clib, clevel, use_dict, filter_flags, nthreads,
                         fp_mantissa_bits, self._eval_method)

    @property
    def clib(self):
        clibs = ["BloscLZ", "LZ4", "LZ4HC", "Snappy", "Zlib", "Zstd", "Lizard"]
        return clibs[self._clib]

    @property
    def clevel(self):
        return self._clevel

    @property
    def filter_flags(self):
        flags = {0: "NOFILTER", 1: "SHUFFLE", 2: "BITSHUFFLE", 4: "DELTA", 8: "TRUNC_PREC"}
        return flags[self._filter_flags]

    @property
    def nthreads(self):
        return self._nthreads

    @property
    def fp_mantissa_bits(self):
        return self._fp_mantissa_bits

    @property
    def filename(self):
        return self._filename

    @property
    def eval_method(self):
        return self._eval_method

    @property
    def seed(self):
        return self._seed

    def __str__(self):
        return (
            "IArray Config object:\n"
            f"    Compression library: {self.clib}\n"
            f"    Compression level: {self.clevel}\n"
            f"    Filter flags: {self.filter_flags}\n"
            f"    Number of threads: {self.nthreads}\n"
            f"    Floating point mantissa bits: {self.fp_mantissa_bits}\n"
            f"    Blocksize: {self.blocksize}\n"
            f"    Filename: {self.filename}\n"
            f"    Eval flags: {self.eval_method}\n"
        )


# TODO: add docstrings
class StorageProperties:

    def __init__(self, backend=ia.BACKEND_BLOSC, chunkshape=None, blockshape=None, enforce_frame=False, filename=None):
        if backend not in (ia.BACKEND_BLOSC, ia.BACKEND_PLAINBUFFER):
            raise ValueError(f"backend can only be BACKEND_BLOSC or BACKEND_PLAINBUFFER")
        self.backend = backend
        self.enforce_frame = True if filename else enforce_frame
        self.filename = filename
        self.chunkshape = chunkshape
        self.blockshape = blockshape

    def get_shape_advice(self, dtshape):
        if self.backend == ia.BACKEND_PLAINBUFFER:
            return
        chunkshape, blockshape = self.chunkshape, self.blockshape
        if chunkshape is not None and blockshape is not None:
            return
        if chunkshape is None and blockshape is None:
            chunkshape_, blockshape_ = ia.partition_advice(dtshape)
            self.chunkshape = chunkshape_
            self.blockshape = blockshape_
            return
        else:
            raise ValueError("You can either specify both chunkshape and blockshape or none of them.")

    def to_tuple(self):
        StoreProp = namedtuple('store_properties', 'backend chunkshape blockshape enforce_frame filename')
        return StoreProp(self.backend, self.chunkshape, self.blockshape, self.backend, self.enforce_frame, self.filename)


#
# Expresssions
#

def fuse_operands(operands1, operands2):
    new_operands = {}
    dup_operands = {}
    new_pos = len(operands1)
    for k2, v2 in operands2.items():
        try:
            k1 = list(operands1.keys())[list(operands1.values()).index(v2)]
            # The operand is duplicated; keep track of it
            dup_operands[k2] = k1
        except ValueError:
            # The value is not among operands1, so rebase it
            new_op = f"o{new_pos}"
            new_pos += 1
            new_operands[new_op] = operands2[k2]
    return new_operands, dup_operands


def fuse_expressions(expr, new_base, dup_op):
    new_expr = ""
    skip_to_char = 0
    old_base = 0
    for i in range(len(expr)):
        if i < skip_to_char:
            continue
        if expr[i] == 'o':
            try:
                j = expr[i + 1:].index(' ')
            except ValueError:
                j = expr[i + 1:].index(')')
            if expr[i + j] == ')':
                j -= 1
            old_pos = int(expr[i+1:i+j+1])
            old_op = f"o{old_pos}"
            if old_op not in dup_op:
                new_pos = old_base + new_base
                new_expr += f"o{new_pos}"
                old_base += 1
            else:
                new_expr += dup_op[old_op]
            skip_to_char = i + j + 1
        else:
            new_expr += expr[i]
    return new_expr


class RandomContext(ext.RandomContext):

    def __init__(self, **kwargs):
        cfg = Config(**kwargs)
        super().__init__(cfg)


class LazyExpr:

    def __init__(self, new_op):
        value1, op, value2 = new_op
        if value2 is None:
            # ufunc
            if isinstance(value1, LazyExpr):
                self.expression = f"{op}({self.expression})"
            else:
                self.operands = {"o0": value1}
                self.expression = f"{op}(o0)"
            return
        elif op in ("atan2", "pow"):
            self.operands = {"o0": value1, "o1": value2}
            self.expression = f"{op}(o0, o1)"
            return
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
                # This is the very first time that a LazyExpr is formed from two operands
                # that are not LazyExpr themselves
                self.operands = {"o0": value1, "o1": value2}
                self.expression = f"(o0 {op} o1)"

    def update_expr(self, new_op):
        # One of the two operands are LazyExpr instances
        value1, op, value2 = new_op
        if isinstance(value1, LazyExpr) and isinstance(value2, LazyExpr):
            # Expression fusion
            # Fuse operands in expressions and detect duplicates
            new_op, dup_op = fuse_operands(value1.operands, value2.operands)
            # Take expression 2 and rebase the operands while removing duplicates
            new_expr = fuse_expressions(value2.expression, len(value1.operands), dup_op)
            self.expression = f"({self.expression} {op} {new_expr})"
            self.operands.update(new_op)
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

    def eval(self, method="iarray_eval", dtype=None, **kwargs):
        # TODO: see if shape and chunkshape can be instance variables, or better stay like this
        o0 = self.operands['o0']
        shape_ = o0.shape

        cfg = Config(**kwargs)
        # TODO: figure out a better way to set a default for the dtype
        dtype = o0.dtype if dtype is None else dtype
        if method == "iarray_eval":
            expr = Expr(**kwargs)
            for k, v in self.operands.items():
                if isinstance(v, IArray):
                    expr.bind(k, v)

            dtshape = ia.dtshape(shape_, dtype)
            cfg._storage.get_shape_advice(dtshape)
            expr.bind_out_properties(dtshape, cfg._storage)
            expr.compile(self.expression)
            out = expr.eval()

        elif method == "numexpr":
            chunkshape = shape_ if cfg._storage.chunkshape is None else cfg._storage.chunkshape
            out = ia.empty(ia.dtshape(shape=shape_, dtype=dtype), **kwargs)
            operand_iters = tuple(o.iter_read_block(chunkshape)
                                  for o in self.operands.values()
                                  if isinstance(o, IArray))
            # put the iterator for the output at the end
            all_iters = operand_iters + (out.iter_write_block(chunkshape),)
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

    def copy(self, view=False, **kwargs):
        cfg = Config(**kwargs)  # chunkshape and blockshape can be passed in storage kwarg
        cfg._storage.get_shape_advice(self.dtshape)
        print(cfg._storage.chunkshape, cfg._storage.blockshape)
        return ext.copy(cfg, self, view)

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

    # def __array_function__(self, func, types, args, kwargs):
    #     if not all(issubclass(t, np.ndarray) for t in types):
    #         # Defer to any non-subclasses that implement __array_function__
    #         return NotImplemented
    #
    #     # Use NumPy's private implementation without __array_function__
    #     # dispatching
    #     return func._implementation(*args, **kwargs)

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    #     print("method:", method)

    def abs(self):
        return LazyExpr(new_op=(self, 'abs', None))

    def arccos(self):
        return LazyExpr(new_op=(self, 'acos', None))

    def arcsin(self):
        return LazyExpr(new_op=(self, 'asin', None))

    def arctan(self):
        return LazyExpr(new_op=(self, 'atan', None))

    def arctan2(self, op2):
        return LazyExpr(new_op=(self, 'atan2', op2))

    def ceil(self):
        return LazyExpr(new_op=(self, 'ceil', None))

    def cos(self):
        return LazyExpr(new_op=(self, 'cos', None))

    def cosh(self):
        return LazyExpr(new_op=(self, 'cosh', None))

    def exp(self):
        return LazyExpr(new_op=(self, 'exp', None))

    def floor(self):
        return LazyExpr(new_op=(self, 'floor', None))

    def log(self):
        return LazyExpr(new_op=(self, 'log', None))

    def log10(self):
        return LazyExpr(new_op=(self, 'log10', None))

    def negative(self):
        return LazyExpr(new_op=(self, 'negate', None))

    def power(self, op2):
        return LazyExpr(new_op=(self, 'pow', op2))

    def sin(self):
        return LazyExpr(new_op=(self, 'sin', None))

    def sinh(self):
        return LazyExpr(new_op=(self, 'sinh', None))

    def sqrt(self):
        return LazyExpr(new_op=(self, 'sqrt', None))

    def tan(self):
        return LazyExpr(new_op=(self, 'tan', None))

    def tanh(self):
        return LazyExpr(new_op=(self, 'tanh', None))


# The main expression class
class Expr(ext.Expression):

    def __init__(self, **kwargs):
        cfg = Config(**kwargs)
        super().__init__(cfg)

    def bind_out_properties(self, dtshape, storage=None):
        if storage is None:
            # Create a default storage
            storage = StorageProperties()
            storage.get_shape_advice(dtshape)
        if storage.chunkshape is None or storage.blockshape is None:
            storage.get_shape_advice(dtshape)
        super().bind_out_properties(dtshape, storage)


#
# Constructors
#

def empty(dtshape, **kwargs):
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.empty(cfg, dtshape)


def arange(dtshape, start=None, stop=None, step=None, **kwargs):
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)

    if (start, stop, step) == (None, None, None):
        stop = np.prod(dtshape.shape)
        start = 0
        step = 1
    elif (stop, step) == (None, None):
        stop = start
        start = 0
        step = 1
    elif step is None:
        stop = stop
        start = start
        if dtshape.shape is None:
            step = 1
        else:
            step = (stop - start) / np.prod(dtshape.shape)

    slice_ = slice(start, stop, step)
    return ext.arange(cfg, slice_, dtshape)


def linspace(dtshape, start, stop, nelem=50, **kwargs):
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)

    shape, dtype = dtshape.to_tuple()
    nelem = np.prod(shape) if dtshape is not None else nelem

    return ext.linspace(cfg, nelem, start, stop, dtshape)


def zeros(dtshape, **kwargs):
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.zeros(cfg, dtshape)


def ones(dtshape, **kwargs):
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.ones(cfg, dtshape)


def full(dtshape, fill_value, **kwargs):
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.full(cfg, fill_value, dtshape)


def save(c, filename, **kwargs):
    cfg = Config(**kwargs)
    return ext.save(cfg, c, filename)


def load(filename, load_in_mem=False, **kwargs):
    cfg = Config(**kwargs)
    return ext.load(cfg, filename, load_in_mem)


def iarray2numpy(iarr, **kwargs):
    cfg = Config(**kwargs)
    return ext.iarray2numpy(cfg, iarr)


def numpy2iarray(c, **kwargs):
    cfg = Config(**kwargs)

    if c.dtype == np.float64:
        dtype = np.float64
    elif c.dtype == np.float32:
        dtype = np.float32
    else:
        raise NotImplementedError("Only float32 and float64 types are supported for now")

    dtshape = ia.dtshape(c.shape, dtype)
    cfg._storage.get_shape_advice(dtshape)
    return ext.numpy2iarray(cfg, c, dtshape)

def random_set_seed(seed):
    ia.RANDOM_SEED = seed

def random_pre(**kwargs):
    ia.RANDOM_SEED += 1
    kwargs["seed"] = ia.RANDOM_SEED
    return kwargs

def random_rand(dtshape, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_rand(cfg, dtshape)


def random_randn(dtshape, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_randn(cfg, dtshape)


def random_beta(dtshape, alpha, beta, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_beta(cfg, alpha, beta, dtshape)


def random_lognormal(dtshape, mu, sigma, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_lognormal(cfg, mu, sigma, dtshape)


def random_exponential(dtshape, beta, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_exponential(cfg, beta, dtshape)


def random_uniform(dtshape, a, b, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_uniform(cfg, a, b, dtshape)


def random_normal(dtshape, mu, sigma, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_normal(cfg, mu, sigma, dtshape)


def random_bernoulli(dtshape, p, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_bernoulli(cfg, p, dtshape)


def random_binomial(dtshape, m, p, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_binomial(cfg, m, p, dtshape)


def random_poisson(dtshape, lamb, **kwargs):
    kwargs = random_pre(**kwargs)
    cfg = Config(**kwargs)
    cfg._storage.get_shape_advice(dtshape)
    return ext.random_poisson(cfg, lamb, dtshape)


def random_kstest(a, b, **kwargs):
    cfg = Config(**kwargs)
    return ext.random_kstest(cfg, a, b)


def matmul(a, b, block_a, block_b, **kwargs):
    cfg = Config(**kwargs)
    return ext.matmul(cfg, a, b, block_a, block_b)


def abs(iarr):
    return iarr.abs()


def arccos(iarr):
    return iarr.arccos()


def arcsin(iarr):
    return iarr.arcsin()


def arctan(iarr):
    return iarr.arctan()


def arctan2(iarr1, iarr2):
    return iarr1.arctan2(iarr2)


def ceil(iarr):
    return iarr.ceil()


def cos(iarr):
    return iarr.cos()


def cosh(iarr):
    return iarr.cosh()


def exp(iarr):
    return iarr.exp()


def floor(iarr):
    return iarr.floor()


def log(iarr):
    return iarr.log()


def log10(iarr):
    return iarr.log10()


def negative(iarr):
    return iarr.negative()


def power(iarr1, iarr2):
    return iarr1.power(iarr2)


def sin(iarr):
    return iarr.sin()


def sinh(iarr):
    return iarr.sinh()


def sqrt(iarr):
    return iarr.sqrt()


def tan(iarr):
    return iarr.tan()


def tanh(iarr):
    return iarr.tanh()


if __name__ == "__main__":
    # Create initial containers
    shape = ia.dtshape([40, 20])
    a1 = ia.linspace(shape, 0, 10)

    # Evaluate with different methods
    a3 = a1.sin() + 2 * a1 + 1
    print(a3)
    a3 += 2
    print(a3)
    a3_np = np.sin(ia.iarray2numpy(a1)) + 2 * ia.iarray2numpy(a1) + 1 + 2
    # a4 = a3.eval(method="numexpr")
    a4 = a3.eval(method="iarray_eval")
    a4_np = ia.iarray2numpy(a4)
    print(a4_np)
    np.testing.assert_allclose(a3_np, a4_np)
