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

import iarray as ia
from iarray import iarray_ext as ext
from itertools import zip_longest
import numpy as np


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
        cfg : Config
            The configuration for this operation.  If None (default), the current
            configuration will be used.
        kwargs : dict
            A dictionary for setting some or all of the fields in the Config
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

    def __str__(self):
        return f"<IArray {self.shape} np.{str(np.dtype(self.dtype))}>"

    def __repr__(self):
        return str(self)

    def __matmul__(self, value):
        a = self
        return ia.matmul(a, value)

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
        return ia.transpose(self, **kwargs)

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


def abs(iarr: IArray):
    return iarr.abs()


def arccos(iarr: IArray):
    return iarr.arccos()


def arcsin(iarr: IArray):
    return iarr.arcsin()


def arctan(iarr: IArray):
    return iarr.arctan()


def arctan2(iarr1: IArray, iarr2: IArray):
    return iarr1.arctan2(iarr2)


def ceil(iarr: IArray):
    return iarr.ceil()


def cos(iarr: IArray):
    return iarr.cos()


def cosh(iarr: IArray):
    return iarr.cosh()


def exp(iarr: IArray):
    return iarr.exp()


def floor(iarr: IArray):
    return iarr.floor()


def log(iarr: IArray):
    return iarr.log()


def log10(iarr: IArray):
    return iarr.log10()


def negative(iarr: IArray):
    return iarr.negative()


def power(iarr1: IArray, iarr2: IArray):
    return iarr1.power(iarr2)


def sin(iarr: IArray):
    return iarr.sin()


def sinh(iarr: IArray):
    return iarr.sinh()


def sqrt(iarr: IArray):
    return iarr.sqrt()


def tan(iarr: IArray):
    return iarr.tan()


def tanh(iarr: IArray):
    return iarr.tanh()

# Reductions

def reduce(a, method, axis=0, cfg=None, **kwargs):
    shape = tuple([s for i, s in enumerate(a.shape) if i != axis])
    dtshape = ia.DTShape(shape, a.dtype)
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.reduce(cfg, a, method, axis)


def max(a, axis=None, cfg=None, **kwargs):
    return reduce(a, ia.Reduce.MAX, axis, cfg, **kwargs)


def min(a, axis=None, cfg=None, **kwargs):
    return reduce(a, ia.Reduce.MIN, axis, cfg, **kwargs)


def sum(a, axis=None, cfg=None, **kwargs):
    return reduce(a, ia.Reduce.SUM, axis, cfg, **kwargs)


def prod(a, axis=None, cfg=None, **kwargs):
    return reduce(a, ia.Reduce.PROD, axis, cfg, **kwargs)


def mean(a, axis=None, cfg=None, **kwargs):
    return reduce(a, ia.Reduce.MEAN, axis, cfg, **kwargs)



def matmul(a: IArray, b: IArray, cfg=None, **kwargs):
    """Multiply two matrices.

    Parameters
    ----------
    a : IArray
        First array.
    b : IArray
        Second array.
    cfg : Config
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config
        dataclass that should override the current configuration.

    Returns
    -------
    IArray
        The resulting array.

    """
    shape = (a.shape[0], b.shape[1]) if b.ndim == 2 else (a.shape[0],)
    dtshape = ia.DTShape(shape, a.dtype)
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.matmul(cfg, a, b)


def transpose(a: IArray, cfg=None, **kwargs):
    """Transpose an array.

    Parameters
    ----------
    a : IArray
        The array to transpose.
    cfg : Config
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config
        dataclass that should override the current configuration.

    Returns
    -------
    IArray
        The transposed array.

    """
    if a.ndim != 2:
        raise AttributeError("Array dimension must be 2")

    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.transpose(cfg, a)
