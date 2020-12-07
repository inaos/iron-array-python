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
from typing import Union
import ndindex
import sys


def is_documented_by(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


# For avoiding a warning in PyCharm in method signatures
IArray = None


class IArray(ext.Container):
    """The ironArray data container.

    This is not meant to be called from user space.
    """

    @property
    def data(self):
        """
        ndarray with array data.
        """
        return ia.iarray2numpy(self)

    def copy(self, view=False, cfg=None, **kwargs) -> IArray:
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

        Returns
        -------
        IArray
            The copy.
        """
        with ia.config(dtshape=self.dtshape, cfg=cfg, **kwargs) as cfg:
            return ext.copy(cfg, self, view)

    def copyto(self, dest):
        """Copy array contents to `dest`.

        Parameters
        ----------
        dest : Any
            The destination container.  It can be any object that supports
            multidimensional assignment (NumPy, Zarr, HDF5...).  It should have the same
            shape than `self`.
        """
        if tuple(dest.shape) != self.shape:
            raise IndexError("Incompatible destination shape")
        for info, block in self.iter_read_block():
            dest[info.slice] = block[:]

    def iter_read_block(self, iterblock: tuple = None):
        if iterblock is None:
            if self.chunkshape is not None:
                iterblock = self.chunkshape
            else:
                iterblock, _ = ia.partition_advice(self.dtshape)
        return ext.ReadBlockIter(self, iterblock)

    def iter_write_block(self, iterblock=None):
        if iterblock is None:
            if self.chunkshape:
                iterblock = self.chunkshape
            else:
                iterblock, _ = ia.partition_advice(self.dtshape)
        return ext.WriteBlockIter(self, iterblock)

    def __getitem__(self, key):
        # Massage the key a bit so that it is compatible with self.shape
        key = list(ndindex.ndindex(key).expand(self.shape).raw)
        squeeze_mask = tuple(True if isinstance(k, int) else False for k in key)

        for i, k in enumerate(key):
            if isinstance(k, np.ndarray):
                raise AttributeError("Advance indexing is not supported yet")
            elif isinstance(k, int):
                key[i] = slice(k, k + 1, None)
            elif isinstance(k, slice):
                if k.step is not None and k.step != 1:
                    raise AttributeError("Step indexing is not supported yet")
            else:
                raise AttributeError(f"Type {type(k)} is not supported")

        start = [sl.start for sl in key]
        stop = [sl.stop for sl in key]
        return super().__getitem__([start, stop, squeeze_mask])

    def __iter__(self):
        return self.iter_read_block()

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
        """See :py:obj:`iarray.abs`."""
        return ia.LazyExpr(new_op=(self, "abs", None))

    def arccos(self):
        """See :py:obj:`iarray.arcos`."""
        return ia.LazyExpr(new_op=(self, "acos", None))

    def arcsin(self):
        """See :py:obj:`iarray.arcsin`."""
        return ia.LazyExpr(new_op=(self, "asin", None))

    def arctan(self):
        """See :py:obj:`iarray.arctan`."""
        return ia.LazyExpr(new_op=(self, "atan", None))

    def arctan2(self, op2):
        """See :py:obj:`iarray.arctan2`."""
        return ia.LazyExpr(new_op=(self, "atan2", op2))

    def acos(self):
        """See :py:obj:`iarray.acos`."""
        return ia.LazyExpr(new_op=(self, "acos", None))

    def asin(self):
        """See :py:obj:`iarray.asin`."""
        return ia.LazyExpr(new_op=(self, "asin", None))

    def atan(self):
        """See :py:obj:`iarray.atan`."""
        return ia.LazyExpr(new_op=(self, "atan", None))

    def atan2(self, op2):
        """See :py:obj:`iarray.atan2`."""
        return ia.LazyExpr(new_op=(self, "atan2", op2))

    def ceil(self):
        """See :py:obj:`iarray.ceil`."""
        return ia.LazyExpr(new_op=(self, "ceil", None))

    def cos(self):
        """See :py:obj:`iarray.cos`."""
        return ia.LazyExpr(new_op=(self, "cos", None))

    def cosh(self):
        """See :py:obj:`iarray.cosh`."""
        return ia.LazyExpr(new_op=(self, "cosh", None))

    def exp(self):
        """See :py:obj:`iarray.exp`."""
        return ia.LazyExpr(new_op=(self, "exp", None))

    def floor(self):
        """See :py:obj:`iarray.floor`."""
        return ia.LazyExpr(new_op=(self, "floor", None))

    def log(self):
        """See :py:obj:`iarray.log`."""
        return ia.LazyExpr(new_op=(self, "log", None))

    def log10(self):
        """See :py:obj:`iarray.log10`."""
        return ia.LazyExpr(new_op=(self, "log10", None))

    def negative(self):
        """See :py:obj:`iarray.negative`."""
        return ia.LazyExpr(new_op=(self, "negate", None))

    def power(self, op2):
        """See :py:obj:`iarray.power`."""
        return ia.LazyExpr(new_op=(self, "pow", op2))

    def sin(self):
        """See :py:obj:`iarray.sin`."""
        return ia.LazyExpr(new_op=(self, "sin", None))

    def sinh(self):
        """See :py:obj:`iarray.sinh`."""
        return ia.LazyExpr(new_op=(self, "sinh", None))

    def sqrt(self):
        """See :py:obj:`iarray.sqrt`."""
        return ia.LazyExpr(new_op=(self, "sqrt", None))

    def tan(self):
        """See :py:obj:`iarray.tan`."""
        return ia.LazyExpr(new_op=(self, "tan", None))

    def tanh(self):
        """See :py:obj:`iarray.tanh`."""
        return ia.LazyExpr(new_op=(self, "tanh", None))


def abs(iarr: IArray):
    """
    Absolute value, element-wise.

    Parameters
    ----------
    iarr: IArray
        Input array.

    Returns
    -------
    abs: IArray
        An array containing the absolute value of each element in x.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.abs.html
    """
    return iarr.abs()


def arccos(iarr: IArray):
    """
    Trigonometric inverse cosine, element-wise.

    The inverse of :py:obj:`cos` so that, if $y = \\\\cos(x)$, then $x = \\\\arccos(y)$.

    Parameters
    ----------
    iarr: IArray
        x-coordinate on the unit circle. For real arguments, the domain is $[-1, 1]$.

    Returns
    -------
    angle: IArray
        The angle of the ray intersecting the unit circle at the given x-coordinate in radians
        $[0, \\\\pi]$.

    Notes
    -----
    :py:obj:`arccos` is a multivalued function: for each x there are infinitely many numbers z
    such that $\\\\cos(z) = x$. The convention is to return the angle z whose real part lies in
    &[0, \\\\pi]$.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.arccos.html
    """
    return iarr.arccos()


def arcsin(iarr: IArray):
    """
    Trigonometric inverse sine, element-wise.

    The inverse of :py:obj:`sin` so that, if $y = \\\\sin(x)$, then $x = \\\\arcsin(y)$.

    Parameters
    ----------
    iarr: IArray
        y-coordinate on the unit circle.

    Returns
    -------
    angle: IArray
        The inverse sine of each element in x, in radians and in the closed interval
        $\\\\left[-\\\\frac{\\\\pi}{2}, \\\\frac{\\\\pi}{2}\\\\right]$.

    Notes
    -----
    :py:obj:`arcsin` is a multivalued function: for each x there are infinitely many numbers z
    such that $\\\\sin(z) = x$. The convention is to return the angle z whose real part lies in
    $\\\\left[-\\\\frac{\\\\pi}{2}, \\\\frac{\\\\pi}{2}\\\\right]$.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html
    """
    return iarr.arcsin()


def arctan(iarr: IArray):
    """
    Trigonometric inverse tangent, element-wise.

    The inverse of :py:obj:`tan` so that, if $y = \\\\tan(x)$, then $x = \\\\arctan(y)$.

    Parameters
    ----------
    iarr: IArray
        Input array.

    Returns
    -------
    angle: IArray
        Array of angles in radians, in the range
        $\\\\left[-\\\\frac{\\\\pi}{2}, \\\\frac{\\\\pi}{2}\\\\right]$.

    Notes
    -----
    :py:obj:`arctan` is a multi-valued function: for each x there are infinitely many numbers z
    such that $\\\\tan(z) = x$. The convention is to return the angle z whose real part lies in
    $\\\\left[-\\\\frac{\\\\pi}{2}, \\\\frac{\\\\pi}{2}\\\\right]$.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.arctan.html
    """

    return iarr.arctan()


def arctan2(iarr1: IArray, iarr2: IArray):
    """
    Element-wise arc tangent of $\\\\frac{iarr_1}{iarr_2}$ choosing the quadrant correctly.


    Parameters
    ----------
    iarr1: IArray
        y-coordinates.
    iarr2: IArray
        x-coordinates.

    Returns
    -------
    angle: IArray
        Array of angles in radians, in the range $[-\\\\pi, \\\\pi]$.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
    """
    return iarr1.arctan2(iarr2)


def ceil(iarr: IArray):
    """
    Return the ceiling of the input, element-wise.  It is often denoted as $\\\\lceil x \\\\rceil$.

    Parameters
    ----------
    iarr: IArray
        Input array.

    Returns
    -------
    out: IArray
        The ceiling of each element in x.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.ceil.html
    """
    return iarr.ceil()


def cos(iarr: IArray):
    """
    Trigonometric cosine, element-wise.

    Parameters
    ----------
    iarr: IArray
        Angle, in radians.

    Returns
    -------
    out: IArray
        The corresponding cosine values.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.cos.html
    """
    return iarr.cos()


def cosh(iarr: IArray):
    """
    Hyperbolic cosine, element-wise.

    Equivalent to ``1/2 * (ia.exp(x) + ia.exp(-x))``.

    Parameters
    ----------
    iarr: IArray
        Input data.

    Returns
    -------
    out: IArray
        The corresponding hyperbolic cosine values.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.cosh.html
    """
    return iarr.cosh()


def exp(iarr: IArray):
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    iarr: IArray
        Input array.

    Returns
    -------
    out: IArray
        Element-wise exponential of input data.

    References
    ----------
    See https://numpy.org/doc/stable/reference/generated/numpy.exp.html
    """
    return iarr.exp()


def floor(iarr: IArray):
    """
    Return the floor of the input, element-wise. It is often denoted as $\\\\lfloor x \\\\rfloor$.

    Parameters
    ----------
    iarr: IArray
        Input array.

    Returns
    -------
    out: IArray
        The floor of each element in input data.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.floor.html
    """
    return iarr.floor()


def log(iarr: IArray):
    """
    Natural logarithm, element-wise.

    The natural logarithm log is the inverse of the exponential function, so that
    $\\\\log(\\\\exp(x)) = x$. The natural logarithm is logarithm in base $e$.

    Parameters
    ----------
    iarr: IArray
        Input array.

    Returns
    -------
    out: IArray
        The natural logarithm of input data, element-wise.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.log.html
    """
    return iarr.log()


def log10(iarr: IArray):
    """
    Return the base 10 logarithm of the input array, element-wise.

    Parameters
    ----------
    iarr: IArray
        Input array.

    Returns
    -------
    out: IArray
        The logarithm to the base 10 of input data, element-wise.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.log10.html
    """
    return iarr.log10()


def negative(iarr: IArray):
    """
    Numerical negative, element-wise.

    Parameters
    ----------
    iarr: IArray
        Input array.

    Returns
    -------
    out: IArray
        Returned array $out = -iarr$.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.log10.html
    """
    return iarr.negative()


def power(iarr1: IArray, iarr2: IArray):
    """
    First array elements raised to powers from second array, element-wise.

    Parameters
    ----------
    iarr1: IArray
        The bases.
    iarr1: IArray
        The exponents.

    Returns
    -------
    out: IArray
        The bases raised to the exponents.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.log10.html"""
    return iarr1.power(iarr2)


def sin(iarr: IArray):
    """
    Trigonometric sine, element-wise.

    Parameters
    ----------
    iarr: IArray
        Angle, in radians.

    Returns
    -------
    out: IArray
        The corresponding sine values.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.sin.html
    """
    return iarr.sin()


def sinh(iarr: IArray):
    """
    Hyperbolic sine, element-wise.

    Equivalent to ``1/2 * (ia.exp(x) - ia.exp(-x))``.

    Parameters
    ----------
    iarr: IArray
        Input data.

    Returns
    -------
    out: IArray
        The corresponding hyperbolic sine values.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.sinh.html
    """
    return iarr.sinh()


def sqrt(iarr: IArray):
    """
    Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    iarr: IArray
        The values whose square-roots are required.

    Returns
    -------
    out: IArray
        An array containing the positive square-root of each element in input data.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
    """
    return iarr.sqrt()


def tan(iarr: IArray):
    """
    Compute tangent element-wise.

    Equivalent to ``ia.sin(x)/ia.cos(x)`` element-wise.

    Parameters
    ----------
    iarr: IArray
        Input data.

    Returns
    -------
    out: IArray
        The corresponding tangent values.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.tan.html
    """
    return iarr.tan()


def tanh(iarr: IArray):
    """
    Compute hyperbolic tangent element-wise.

    Equivalent to ``ia.sinh(x)/ia.cosh(x)``.

    Parameters
    ----------
    iarr: IArray
        Input data.

    Returns
    -------
    out: IArray
        The corresponding hyperbolic tangent values.

    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.tanh.html
    """
    return iarr.tanh()


# Reductions


def reduce(
    a: IArray, method: ia.Reduce, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs
):
    if axis is None:
        axis = range(a.ndim)
    if isinstance(axis, int):
        axis = (axis,)

    shape = tuple([s for i, s in enumerate(a.shape) if i != axis])
    dtshape = ia.DTShape(shape, a.dtype)
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        c = ext.reduce_multi(cfg, a, method, axis)
        if c.ndim == 0:
            c = float(ia.iarray2numpy(c))
        return c


def max(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : IArray
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : Config or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config dataclass that should
        override the current configuration.
    Returns
    -------
    max : IArray or float
        Maximum of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """

    return reduce(a, ia.Reduce.MAX, axis, cfg, **kwargs)


def min(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : IArray
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : Config or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config dataclass that should
        override the current configuration.
    Returns
    -------
    min : IArray or float
        Minimum of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """
    return reduce(a, ia.Reduce.MIN, axis, cfg, **kwargs)


def sum(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Return the sum of array elements over a given axis.

    Parameters
    ----------
    a : IArray
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : Config or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config dataclass that should
        override the current configuration.
    Returns
    -------
    sum : IArray or float
        Sum of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """
    return reduce(a, ia.Reduce.SUM, axis, cfg, **kwargs)


def prod(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : IArray
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : Config or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config dataclass that should
        override the current configuration.
    Returns
    -------
    prod : IArray or float
        Product of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """
    return reduce(a, ia.Reduce.PROD, axis, cfg, **kwargs)


def mean(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Compute the arithmetic mean along the specified axis. Returns the average of the array elements.

    Parameters
    ----------
    a : IArray
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : Config or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config dataclass that should
        override the current configuration.
    Returns
    -------
    mean : IArray or float
        Mean of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """
    return reduce(a, ia.Reduce.MEAN, axis, cfg, **kwargs)


# Linear Algebra


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
