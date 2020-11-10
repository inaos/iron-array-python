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
from iarray import py2llvm
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
    """Shape and data type dataclass.

    Parameters
    ----------
    shape: list, tuple
        The shape of the array.
    dtype: np.float32, np.float64
        The data type of the elements in the array.  The default is np.float64.
    """

    shape: Sequence
    dtype: (np.float32, np.float64) = np.float64

    def __post_init__(self):
        if not self.shape:
            raise ValueError("shape must be non-empty")


#
# Expressions
#


def check_inputs(inputs: list):
    first_input = inputs[0]
    for input in inputs[1:]:
        if first_input.shape != input.shape:
            raise ValueError("Inputs should have the same shape")
        if first_input.dtype != input.dtype:
            raise TypeError("Inputs should have the same dtype")
    return first_input.dtshape


def expr_from_string(sexpr: str, inputs: dict, cfg: ia.Config = None, **kwargs):
    """Create an `Expr` instance from a expression in string form.

    Parameters
    ----------
    sexpr : str
        An expression in string format.
    inputs : dict
        Map of variables in `sexpr` to actual arrays.
    cfg : ia.Config
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the ia.Config
        dataclass that should override the current configuration.

    Returns
    -------
    ia.Expr
        An expression ready to be evaluated via `.eval()`.
    """
    dtshape = check_inputs(list(inputs.values()))
    expr = Expr(dtshape=dtshape, cfg=cfg, **kwargs)
    for i in inputs:
        expr.bind(i, inputs[i])
    expr.compile(sexpr)
    return expr


def expr_from_udf(udf: py2llvm.Function, inputs: list, cfg=None, **kwargs):
    """Create an `Expr` instance from an UDF function.

    Parameters
    ----------
    udf : py2llvm.Function
        A User Defined Function.
    inputs : list
        List of arrays whose values are passed as arguments, after the output,
        to the UDF function.
    cfg : ia.Config
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the ia.Config
        dataclass that should override the current configuration.

    Returns
    -------
    ia.Expr
        An expression ready to be evaluated via `.eval()`.
    """
    dtshape = check_inputs(inputs)
    expr = Expr(dtshape=dtshape, cfg=cfg, **kwargs)
    for i in inputs:
        expr.bind("", i)
    expr.compile_udf(udf)
    return expr


class RandomContext(ext.RandomContext):
    def __init__(self, **kwargs):
        with ia.config(**kwargs) as cfg:
            super().__init__(cfg)


class IArray(ext.Container):
    """The ironArray data container.

    This is not meant to be called from user space.
    """

    def copy(self, view=False, cfg=None, **kwargs):
        """Return a copy of the array.

        Parameters
        ----------
        view : bool
            If True, return a view; else an actual copy.  Default is False.
        cfg : ia.Config
            The configuration for this operation.  If None (default), the current
            configuration will be used.
        kwargs : dict
            A dictionary for setting some or all of the fields in the ia.Config
            dataclass that should override the current configuration.
        """
        with ia.config(dtshape=self.dtshape, cfg=cfg, **kwargs) as cfg:
            return ext.copy(cfg, self, view)

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

    def __matmul__(self, value):
        a = self
        return matmul(a, value)

    def __add__(self, value):
        return ia.LazyExpr(new_op=(self, "+", value))

    def __radd__(self, value):
        return ia.LazyExpr(new_op=(value, "+", self))

    def __sub__(self, value):
        return ia.LazyExpr(new_op=(self, "-", value))

    def __rsub__(self, value):
        return ia.LazyExpr(new_op=(value, "-", self))

    def __mul__(self, value):
        return ia.LazyExpr(new_op=(self, "*", value))

    def __rmul__(self, value):
        return ia.LazyExpr(new_op=(value, "*", self))

    def __truediv__(self, value):
        return ia.LazyExpr(new_op=(self, "/", value))

    def __rtruediv__(self, value):
        return ia.LazyExpr(new_op=(value, "/", self))

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

    @property
    def T(self):
        return self.transpose()

    def transpose(self, **kwargs):
        return transpose(self, **kwargs)

    def abs(self):
        return ia.LazyExpr(new_op=(self, "abs", None))

    def arccos(self):
        return ia.LazyExpr(new_op=(self, "acos", None))

    def arcsin(self):
        return ia.LazyExpr(new_op=(self, "asin", None))

    def arctan(self):
        return ia.LazyExpr(new_op=(self, "atan", None))

    def arctan2(self, op2):
        return ia.LazyExpr(new_op=(self, "atan2", op2))

    def acos(self):
        return ia.LazyExpr(new_op=(self, "acos", None))

    def asin(self):
        return ia.LazyExpr(new_op=(self, "asin", None))

    def atan(self):
        return ia.LazyExpr(new_op=(self, "atan", None))

    def atan2(self, op2):
        return ia.LazyExpr(new_op=(self, "atan2", op2))

    def ceil(self):
        return ia.LazyExpr(new_op=(self, "ceil", None))

    def cos(self):
        return ia.LazyExpr(new_op=(self, "cos", None))

    def cosh(self):
        return ia.LazyExpr(new_op=(self, "cosh", None))

    def exp(self):
        return ia.LazyExpr(new_op=(self, "exp", None))

    def floor(self):
        return ia.LazyExpr(new_op=(self, "floor", None))

    def log(self):
        return ia.LazyExpr(new_op=(self, "log", None))

    def log10(self):
        return ia.LazyExpr(new_op=(self, "log10", None))

    def negative(self):
        return ia.LazyExpr(new_op=(self, "negate", None))

    def power(self, op2):
        return ia.LazyExpr(new_op=(self, "pow", op2))

    def sin(self):
        return ia.LazyExpr(new_op=(self, "sin", None))

    def sinh(self):
        return ia.LazyExpr(new_op=(self, "sinh", None))

    def sqrt(self):
        return ia.LazyExpr(new_op=(self, "sqrt", None))

    def tan(self):
        return ia.LazyExpr(new_op=(self, "tan", None))

    def tanh(self):
        return ia.LazyExpr(new_op=(self, "tanh", None))


# The main expression class
class Expr(ext.Expression):
    """An class that is meant to hold an expression.

    This is not meant to be called directly from user space.

    See Also
    --------
    ia.expr_from_string
    ia.expr_from_udf
    """

    def __init__(self, dtshape, cfg=None, **kwargs):
        with ia.config(cfg=cfg, dtshape=dtshape, **kwargs) as cfg:
            self.cfg = cfg
            super().__init__(self.cfg)
            super().bind_out_properties(dtshape, cfg.storage)

    def eval(self):
        """Evaluate the expression in self.

        Returns
        -------
        ia.IArray
            The output array.
        """
        return super().eval()


#
# Constructors
#


def empty(dtshape, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.empty(cfg, dtshape)


def arange(dtshape, start=None, stop=None, step=None, cfg=None, **kwargs):
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

    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.arange(cfg, slice_, dtshape)


def linspace(dtshape, start, stop, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.linspace(cfg, start, stop, dtshape)


def zeros(dtshape, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.zeros(cfg, dtshape)


def ones(dtshape, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.ones(cfg, dtshape)


def full(dtshape, fill_value, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.full(cfg, fill_value, dtshape)


def save(c, filename, cfg=None, **kwargs):
    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.save(cfg, c, filename)


def load(filename, load_in_mem=False, cfg=None, **kwargs):
    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.load(cfg, filename, load_in_mem)


def iarray2numpy(iarr, cfg=None, **kwargs):
    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.iarray2numpy(cfg, iarr)


def numpy2iarray(c, cfg=None, **kwargs):
    if c.dtype == np.float64:
        dtype = np.float64
    elif c.dtype == np.float32:
        dtype = np.float32
    else:
        raise NotImplementedError("Only float32 and float64 types are supported for now")

    dtshape = ia.DTShape(c.shape, dtype)
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.numpy2iarray(cfg, c, dtshape)


def random_rand(dtshape, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_rand(cfg, dtshape)


def random_randn(dtshape, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_randn(cfg, dtshape)


def random_beta(dtshape, alpha, beta, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_beta(cfg, alpha, beta, dtshape)


def random_lognormal(dtshape, mu, sigma, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_lognormal(cfg, mu, sigma, dtshape)


def random_exponential(dtshape, beta, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_exponential(cfg, beta, dtshape)


def random_uniform(dtshape, a, b, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_uniform(cfg, a, b, dtshape)


def random_normal(dtshape, mu, sigma, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_normal(cfg, mu, sigma, dtshape)


def random_bernoulli(dtshape, p, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_bernoulli(cfg, p, dtshape)


def random_binomial(dtshape, m, p, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_binomial(cfg, m, p, dtshape)


def random_poisson(dtshape, lamb, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_poisson(cfg, lamb, dtshape)


def random_kstest(a, b, cfg=None, **kwargs):
    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.random_kstest(cfg, a, b)


def matmul(a, b, cfg=None, **kwargs):
    shape = (a.shape[0], b.shape[1]) if b.ndim == 2 else (a.shape[0],)
    dtshape = DTShape(shape, a.dtype)
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.matmul(cfg, a, b)


def transpose(a: IArray, cfg=None, **kwargs):
    if a.ndim != 2:
        raise AttributeError("Array dimension must be 2")

    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.transpose(cfg, a)


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
