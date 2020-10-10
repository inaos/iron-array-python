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

import iarray as ia
from iarray import iarray_ext as ext
from itertools import zip_longest
from dataclasses import dataclass
from typing import Sequence


def cmp_arrays(a, b, success=None):
    """Quick and dirty comparison between arrays `a` and `b`.

    The arrays `a` and `b` are converted internally to numpy arrays, so
    this can require a lot of memory.  This is mainly used for quick testing.

    If success, and the string passed in `success` is not None, it is printed.
    """
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


@dataclass(frozen=True)
class DTShape:
    shape: Sequence
    dtype: (np.float32, np.float64) = np.float64


#
# Expressions
#


def create_expr(str_expr, inputs, dtshape, **kwargs):
    """Create an `Expr` instance.

    `str_expr` is the expression in string format.

    `inputs` is a dictionary that maps variables in `str_expr` to actual arrays.

    `dtshape` is a `dtshape` instance with the shape and dtype of the resulting array.

    `**kwargs` can be any argument supported by the `Config()` constructor.  These will
    be used for both the evaluation process and the resulting array.
    """
    expr = ia.Expr(**kwargs)
    for i in inputs:
        expr.bind(i, inputs[i])
    expr.bind_out_properties(dtshape)
    expr.compile(str_expr)
    return expr


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
        if expr[i] == "o":
            try:
                j = expr[i + 1 :].index(" ")
            except ValueError:
                j = expr[i + 1 :].index(")")
            if expr[i + j] == ")":
                j -= 1
            old_pos = int(expr[i + 1 : i + j + 1])
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
        cparams = ia.ConfigParams(**kwargs)
        super().__init__(cparams)


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
                    op_name = list(value1.operands.keys())[
                        list(value1.operands.values()).index(value2)
                    ]
                except ValueError:
                    op_name = f"o{len(self.operands)}"
                    self.operands[op_name] = value2
                self.expression = f"({self.expression} {op} {op_name})"
        else:
            if isinstance(value1, (int, float)):
                self.expression = f"({value1} {op} {self.expression})"
            else:
                try:
                    op_name = list(value2.operands.keys())[
                        list(value2.operands.values()).index(value1)
                    ]
                except ValueError:
                    op_name = f"o{len(self.operands)}"
                    self.operands[op_name] = value1
                self.expression = f"({op_name} {op} {self.expression})"
        return self

    def __add__(self, value):
        return self.update_expr(new_op=(self, "+", value))

    def __radd__(self, value):
        return self.update_expr(new_op=(value, "+", self))

    def __sub__(self, value):
        return self.update_expr(new_op=(self, "-", value))

    def __rsub__(self, value):
        return self.update_expr(new_op=(value, "-", self))

    def __mul__(self, value):
        return self.update_expr(new_op=(self, "*", value))

    def __rmul__(self, value):
        return self.update_expr(new_op=(value, "*", self))

    def __truediv__(self, value):
        return self.update_expr(new_op=(self, "/", value))

    def __rtruediv__(self, value):
        return self.update_expr(new_op=(value, "/", self))

    def eval(self, dtshape, **kwargs):
        cparams = ia.ConfigParams(**kwargs)
        expr = Expr(**kwargs)
        for k, v in self.operands.items():
            if isinstance(v, IArray):
                expr.bind(k, v)

        cparams.storage.get_shape_advice(dtshape)
        expr.bind_out_properties(dtshape, cparams.storage)
        expr.compile(self.expression)
        out = expr.eval()

        return out

    def __str__(self):
        expression = f"{self.expression}"
        return expression


# The main IronArray container (not meant to be called from user space)
class IArray(ext.Container):
    def copy(self, view=False, **kwargs):
        cparams = ia.ConfigParams(
            **kwargs
        )  # chunkshape and blockshape can be passed in storage kwarg
        cparams.storage.get_shape_advice(self.dtshape)
        return ext.copy(cparams, self, view)

    def __getitem__(self, key):
        # Massage the key a bit so that it is compatible with self.shape
        if self.ndim == 1:
            key = [key]
        start = [s.start if s.start is not None else 0 for s in key]
        start = [st if st < sh else sh for st, sh in zip_longest(start, self.shape, fillvalue=0)]
        stop = [
            s.stop if s.stop is not None else sh
            for s, sh in zip_longest(key, self.shape, fillvalue=slice(0))
        ]
        stop = [sh if st == 0 else st for st, sh in zip_longest(stop, self.shape)]
        stop = [st if st < sh else sh for st, sh in zip_longest(stop, self.shape)]

        # Check that the final size is not zero, as this is not supported yet in iArray
        length = 1
        for s0, s1 in zip_longest(start, stop):
            length *= s1 - s0
        if length < 1:
            raise ValueError("Slices with 0 or negative dims are not supported yet")

        return super().__getitem__([start, stop])

    def __add__(self, value):
        return LazyExpr(new_op=(self, "+", value))

    def __radd__(self, value):
        return LazyExpr(new_op=(value, "+", self))

    def __sub__(self, value):
        return LazyExpr(new_op=(self, "-", value))

    def __rsub__(self, value):
        return LazyExpr(new_op=(value, "-", self))

    def __mul__(self, value):
        return LazyExpr(new_op=(self, "*", value))

    def __rmul__(self, value):
        return LazyExpr(new_op=(value, "*", self))

    def __truediv__(self, value):
        return LazyExpr(new_op=(self, "/", value))

    def __rtruediv__(self, value):
        return LazyExpr(new_op=(value, "/", self))

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
        return LazyExpr(new_op=(self, "abs", None))

    def arccos(self):
        return LazyExpr(new_op=(self, "acos", None))

    def arcsin(self):
        return LazyExpr(new_op=(self, "asin", None))

    def arctan(self):
        return LazyExpr(new_op=(self, "atan", None))

    def arctan2(self, op2):
        return LazyExpr(new_op=(self, "atan2", op2))

    def acos(self):
        return LazyExpr(new_op=(self, "acos", None))

    def asin(self):
        return LazyExpr(new_op=(self, "asin", None))

    def atan(self):
        return LazyExpr(new_op=(self, "atan", None))

    def atan2(self, op2):
        return LazyExpr(new_op=(self, "atan2", op2))

    def ceil(self):
        return LazyExpr(new_op=(self, "ceil", None))

    def cos(self):
        return LazyExpr(new_op=(self, "cos", None))

    def cosh(self):
        return LazyExpr(new_op=(self, "cosh", None))

    def exp(self):
        return LazyExpr(new_op=(self, "exp", None))

    def floor(self):
        return LazyExpr(new_op=(self, "floor", None))

    def log(self):
        return LazyExpr(new_op=(self, "log", None))

    def log10(self):
        return LazyExpr(new_op=(self, "log10", None))

    def negative(self):
        return LazyExpr(new_op=(self, "negate", None))

    def power(self, op2):
        return LazyExpr(new_op=(self, "pow", op2))

    def sin(self):
        return LazyExpr(new_op=(self, "sin", None))

    def sinh(self):
        return LazyExpr(new_op=(self, "sinh", None))

    def sqrt(self):
        return LazyExpr(new_op=(self, "sqrt", None))

    def tan(self):
        return LazyExpr(new_op=(self, "tan", None))

    def tanh(self):
        return LazyExpr(new_op=(self, "tanh", None))


# The main expression class
class Expr(ext.Expression):
    def __init__(self, **kwargs):
        self.cparams = ia.ConfigParams(**kwargs)
        super().__init__(self.cparams)

    def bind_out_properties(self, dtshape, storage=None):
        if storage is None:
            # Use the default storage in config
            storage = self.cparams.storage
            storage.get_shape_advice(dtshape)
        if storage.chunkshape is None or storage.blockshape is None:
            storage.get_shape_advice(dtshape)
        super().bind_out_properties(dtshape, storage)


#
# Constructors
#


def empty(dtshape, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.empty(cparams, dtshape)


def arange(dtshape, start=None, stop=None, step=None, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)

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
    return ext.arange(cparams, slice_, dtshape)


def linspace(dtshape, start, stop, nelem=50, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)

    nelem = np.prod(dtshape.shape) if dtshape is not None else nelem

    return ext.linspace(cparams, nelem, start, stop, dtshape)


def zeros(dtshape, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.zeros(cparams, dtshape)


def ones(dtshape, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.ones(cparams, dtshape)


def full(dtshape, fill_value, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.full(cparams, fill_value, dtshape)


def save(c, filename, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    return ext.save(cparams, c, filename)


def load(filename, load_in_mem=False, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    return ext.load(cparams, filename, load_in_mem)


def iarray2numpy(iarr, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    return ext.iarray2numpy(cparams, iarr)


def numpy2iarray(c, **kwargs):
    cparams = ia.ConfigParams(**kwargs)

    if c.dtype == np.float64:
        dtype = np.float64
    elif c.dtype == np.float32:
        dtype = np.float32
    else:
        raise NotImplementedError("Only float32 and float64 types are supported for now")

    dtshape = ia.DTShape(c.shape, dtype)
    cparams.storage.get_shape_advice(dtshape)
    return ext.numpy2iarray(cparams, c, dtshape)


def random_set_seed(seed):
    ia.RANDOM_SEED = seed


def random_pre(**kwargs):
    ia.RANDOM_SEED += 1
    kwargs["seed"] = ia.RANDOM_SEED
    return kwargs


def random_rand(dtshape, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_rand(cparams, dtshape)


def random_randn(dtshape, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_randn(cparams, dtshape)


def random_beta(dtshape, alpha, beta, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_beta(cparams, alpha, beta, dtshape)


def random_lognormal(dtshape, mu, sigma, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_lognormal(cparams, mu, sigma, dtshape)


def random_exponential(dtshape, beta, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_exponential(cparams, beta, dtshape)


def random_uniform(dtshape, a, b, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_uniform(cparams, a, b, dtshape)


def random_normal(dtshape, mu, sigma, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_normal(cparams, mu, sigma, dtshape)


def random_bernoulli(dtshape, p, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_bernoulli(cparams, p, dtshape)


def random_binomial(dtshape, m, p, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_binomial(cparams, m, p, dtshape)


def random_poisson(dtshape, lamb, **kwargs):
    kwargs = random_pre(**kwargs)
    cparams = ia.ConfigParams(**kwargs)
    cparams.storage.get_shape_advice(dtshape)
    return ext.random_poisson(cparams, lamb, dtshape)


def random_kstest(a, b, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    return ext.random_kstest(cparams, a, b)


def matmul(a, b, block_a, block_b, **kwargs):
    cparams = ia.ConfigParams(**kwargs)
    return ext.matmul(cparams, a, b, block_a, block_b)


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
    # Check representations of default configs
    print(ia.ConfigParams())
    print(ia.Storage())

    print()
    # Create initial containers
    dtshape_ = ia.DTShape([40, 20])
    a1 = ia.linspace(dtshape_, 0, 10)

    # Evaluate with different methods
    a3 = a1.sin() + 2 * a1 + 1
    print(a3)
    a3 += 2
    # print(a3)
    a3_np = np.sin(ia.iarray2numpy(a1)) + 2 * ia.iarray2numpy(a1) + 1 + 2
    a4 = a3.eval(dtshape_)
    a4_np = ia.iarray2numpy(a4)
    # print(a4_np)
    np.testing.assert_allclose(a3_np, a4_np)
    print("Everything is working fine")
