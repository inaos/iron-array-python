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
import numpy as np
from typing import Union
import ndindex
from .info import InfoReporter


def process_key(key, shape):
    key = ndindex.ndindex(key).expand(shape).raw
    mask = tuple(True if isinstance(k, int) else False for k in key)
    key = tuple(k if isinstance(k, slice) else slice(k, k + 1, None) for k in key)
    return key, mask


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
    def info(self):
        """
        Print information about this array.
        """
        return InfoReporter(self)

    @property
    def info_items(self):
        items = []
        items += [("type", self.__class__.__name__)]
        items += [("shape", self.shape)]
        items += [("chunks", self.chunks)]
        items += [("blocks", self.blocks)]
        items += [("cratio", f"{self.cratio:.2f}")]
        return items

    @property
    def data(self):
        """
        Get a ndarray with array data.

        Returns
        -------
        out: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        """
        return ia.iarray2numpy(self)

    def copy(self, cfg=None, **kwargs) -> IArray:
        """Return a copy of the array.

        Parameters
        ----------
        cfg : :class:`Config`
            The configuration for this operation.  If None (default), the
            configuration from self will be used instead of that of the current configuration.
        kwargs : dict
            A dictionary for setting some or all of the fields in the :class:`Config`
            dataclass that should override the configuration.

        Returns
        -------
        :ref:`IArray`
            The copy.
        """
        if cfg is None:
            cfg = ia.get_config(self.cfg)
            # the urlpath should not be copied
            cfg.store.urlpath = None
            cfg.urlpath = None

        # Generally we don't want btune to optimize, except if specified
        btune = False
        if "btune" in kwargs:
            btune = kwargs["btune"]
            kwargs.pop("btune")
        with ia.config(shape=self.shape, cfg=cfg, btune=btune, **kwargs) as cfg:
            return ext.copy(cfg, self)

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
            if self.chunks is not None:
                iterblock = self.chunks
            else:
                iterblock, _ = ia.partition_advice(self.shape)
        return ext.ReadBlockIter(self, iterblock)

    def iter_write_block(self, iterblock=None):
        if iterblock is None:
            if self.chunks:
                iterblock = self.chunks
            else:
                iterblock, _ = ia.partition_advice(self.shape)
        return ext.WriteBlockIter(self, iterblock)

    def __getitem__(self, key):
        # Massage the key a bit so that it is compatible with self.shape
        key, mask = process_key(key, self.shape)
        start = [sl.start for sl in key]
        stop = [sl.stop for sl in key]

        return super().__getitem__([start, stop, mask])

    def __setitem__(self, key, value):
        key, mask = process_key(key, self.shape)
        start = [sl.start for sl in key]
        stop = [sl.stop for sl in key]

        shape = [sp - st for sp, st in zip(stop, start)]
        if isinstance(value, (float, int)):
            value = np.full(shape, value, dtype=self.dtype)
        elif isinstance(value, ia.IArray):
            value = value.data
        with ia.config(cfg=self.cfg) as cfg:
            return ext.set_slice(cfg, self, start, stop, value)

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
        """Transpose the array.

        Parameters
        ----------
        kwargs : dict
            A dictionary for setting some or all of the fields in the :class:`Config`
            dataclass that should override the current configuration.

        Returns
        -------
        :ref:`IArray`
            The transposed array.

        """
        return ia.transpose(self, **kwargs)

    def abs(self):
        """
        Absolute value, element-wise.

        Parameters
        ----------
        iarr: :ref:`IArray`
           Input array.

        Returns
        -------
        abs: :ref:`IArray`
           An array containing the absolute value of each element in x.

        References
        ----------
        `np.absolute <https://numpy.org/doc/stable/reference/generated/numpy.absolute.html>`_
        """
        return ia.LazyExpr(new_op=(self, "abs", None))

    def arccos(self):
        """
        Trigonometric inverse cosine, element-wise.

        The inverse of :py:obj:`cos` so that, if :math:`y = \\cos(x)`, then :math:`x = \\arccos(y)`.

        Parameters
        ----------
        iarr: :ref:`IArray`
            x-coordinate on the unit circle. For real arguments, the domain is :math:`\\left [ -1, 1 \\right]`.

        Returns
        -------
        angle: :ref:`IArray`
            The angle of the ray intersecting the unit circle at the given x-coordinate in radians
            :math:`[0, \\pi]`.

        Notes
        -----
        :py:obj:`arccos` is a multivalued function: for each :math:`x` there are infinitely many numbers :math:`z`
        such that :math:`\\cos(z) = x`. The convention is to return the angle :math:`z` whose real part lies in
        :math:`\\left [ 0, \\pi \\right]`.

        References
        ----------
        `np.arccos <https://numpy.org/doc/stable/reference/generated/numpy.arccos.html>`_
        """
        return ia.LazyExpr(new_op=(self, "acos", None))

    def arcsin(self):
        """
        Trigonometric inverse sine, element-wise.

        The inverse of :py:obj:`sin` so that, if :math:`y = \\sin(x)`, then :math:`x = \\arcsin(y)`.

        Parameters
        ----------
        iarr: :ref:`IArray`
           y-coordinate on the unit circle.

        Returns
        -------
        angle: :ref:`IArray`
           The inverse sine of each element in :math:`x`, in radians and in the closed interval
           :math:`\\left[-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right]`.

        Notes
        -----
        :py:obj:`arcsin` is a multivalued function: for each :math:`x` there are infinitely many numbers :math:`z`
        such that :math:`\\sin(z) = x`. The convention is to return the angle :math:`z` whose real part lies in
        :math:`\\left[-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right]`.

        References
        ----------
        `np.arcsin <https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html>`_
        """
        return ia.LazyExpr(new_op=(self, "asin", None))

    def arctan(self):
        """
        Trigonometric inverse tangent, element-wise.

        The inverse of :py:obj:`tan` so that, if :math:`y = \\tan(x)`, then :math:`x = \\arctan(y)`.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input array.

        Returns
        -------
        angle: :ref:`IArray`
            Array of angles in radians, in the range
            :math:`\\left[-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right]`.

        Notes
        -----
        :py:obj:`arctan` is a multi-valued function: for each x there are infinitely many numbers :math:`z`
        such that :math:`\\tan(z) = x`. The convention is to return the angle :math:`z` whose real part lies in
        :math:`\\left[-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right]`.

        References
        ----------
        `np.arctan <https://numpy.org/doc/stable/reference/generated/numpy.arctan.html>`_
        """
        return ia.LazyExpr(new_op=(self, "atan", None))

    def arctan2(self, op2):
        """
        Element-wise arc tangent of :math:`\\frac{iarr_1}{iarr_2}` choosing the quadrant correctly.


        Parameters
        ----------
        iarr1: :ref:`IArray`
            y-coordinates.
        iarr2: :ref:`IArray`
            x-coordinates.

        Returns
        -------
        angle: :ref:`IArray`
            Array of angles in radians, in the range :math:`[-\\pi, \\pi]`.

        References
        ----------
        `np.arctan2 <https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html>`_
        """
        return ia.LazyExpr(new_op=(self, "atan2", op2))

    def acos(self):
        """See :py:obj:`IArray.arccos`."""
        return ia.LazyExpr(new_op=(self, "acos", None))

    def asin(self):
        """See :py:obj:`IArray.arcsin`."""
        return ia.LazyExpr(new_op=(self, "asin", None))

    def atan(self):
        """See :py:obj:`IArray.arctan`."""
        return ia.LazyExpr(new_op=(self, "atan", None))

    def atan2(self, op2):
        """See :py:obj:`IArray.arctan2`."""
        return ia.LazyExpr(new_op=(self, "atan2", op2))

    def ceil(self):
        """
        Return the ceiling of the input, element-wise.  It is often denoted as :math:`\\lceil x \\rceil`.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`IArray`
            The ceiling of each element in :math:`x`.

        References
        ----------
        `np.ceil <https://numpy.org/doc/stable/reference/generated/numpy.ceil.html>`_
        """
        return ia.LazyExpr(new_op=(self, "ceil", None))

    def cos(self):
        """
        Trigonometric cosine, element-wise.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Angle, in radians.

        Returns
        -------
        out: :ref:`IArray`
            The corresponding cosine values.

        References
        ----------
        `np.cos <https://numpy.org/doc/stable/reference/generated/numpy.cos.html>`_
        """
        return ia.LazyExpr(new_op=(self, "cos", None))

    def cosh(self):
        """
        Hyperbolic cosine, element-wise.

        Equivalent to ``1/2 * (ia.exp(x) + ia.exp(-x))``.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input data.

        Returns
        -------
        out: :ref:`IArray`
            The corresponding hyperbolic cosine values.

        References
        ----------
        `np.cosh <https://numpy.org/doc/stable/reference/generated/numpy.cosh.html>`_
        """
        return ia.LazyExpr(new_op=(self, "cosh", None))

    def exp(self):
        """
        Calculate the exponential of all elements in the input array.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`IArray`
            Element-wise exponential of input data.

        References
        ----------
        `np.exp <https://numpy.org/doc/stable/reference/generated/numpy.exp.html>`_
        """
        return ia.LazyExpr(new_op=(self, "exp", None))

    def floor(self):
        """
        Return the floor of the input, element-wise. It is often denoted as :math:`\\lfloor x \\rfloor`.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`IArray`
            The floor of each element in input data.

        References
        ----------
        `np.floor <https://numpy.org/doc/stable/reference/generated/numpy.floor.html>`_
        """
        return ia.LazyExpr(new_op=(self, "floor", None))

    def log(self):
        """
        Natural logarithm, element-wise.

        The natural logarithm log is the inverse of the exponential function, so that
        :math:`\\log(\\exp(x)) = x`. The natural logarithm is logarithm in base :math:`e`.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`IArray`
            The natural logarithm of input data, element-wise.

        References
        ----------
        `np.log <https://numpy.org/doc/stable/reference/generated/numpy.log.html>`_
        """
        return ia.LazyExpr(new_op=(self, "log", None))

    def log10(self):
        """
        Return the base 10 logarithm of the input array, element-wise.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`IArray`
            The logarithm to the base 10 of input data, element-wise.

        References
        ----------
        `np.log10 <https://numpy.org/doc/stable/reference/generated/numpy.log10.html>`_
        """
        return ia.LazyExpr(new_op=(self, "log10", None))

    def negative(self):
        """
        Numerical negative, element-wise.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`IArray`
            Returned array :math:`out = -iarr`.

        References
        ----------
        `np.negative <https://numpy.org/doc/stable/reference/generated/numpy.negative.html>`_
        """
        return ia.LazyExpr(new_op=(self, "negate", None))

    def power(self, op2):
        """
        First array elements raised to powers from second array, element-wise.

        Parameters
        ----------
        iarr1: :ref:`IArray`
            The bases.
        iarr1: :ref:`IArray`
            The exponents.

        Returns
        -------
        out: :ref:`IArray`
            The bases raised to the exponents.

        References
        ----------
        `np.power <https://numpy.org/doc/stable/reference/generated/numpy.power.html>`_
        """
        return ia.LazyExpr(new_op=(self, "pow", op2))

    def sin(self):
        """
        Trigonometric sine, element-wise.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Angle, in radians.

        Returns
        -------
        out: :ref:`IArray`
            The corresponding sine values.

        References
        ----------
        `np.sin <https://numpy.org/doc/stable/reference/generated/numpy.sin.html>`_
        """
        return ia.LazyExpr(new_op=(self, "sin", None))

    def sinh(self):
        """
        Hyperbolic sine, element-wise.

        Equivalent to ``1/2 * (ia.exp(x) - ia.exp(-x))``.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input data.

        Returns
        -------
        out: :ref:`IArray`
            The corresponding hyperbolic sine values.

        References
        ----------
        `np.sinh <https://numpy.org/doc/stable/reference/generated/numpy.sinh.html>`_
        """
        return ia.LazyExpr(new_op=(self, "sinh", None))

    def sqrt(self):
        """
        Return the non-negative square-root of an array, element-wise.

        Parameters
        ----------
        iarr: :ref:`IArray`
            The values whose square-roots are required.

        Returns
        -------
        out: :ref:`IArray`
            An array containing the positive square-root of each element in input data.

        References
        ----------
        `np.sqrt <https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html>`_
        """
        return ia.LazyExpr(new_op=(self, "sqrt", None))

    def tan(self):
        """
        Compute tangent element-wise.

        Equivalent to ``ia.sin(x)/ia.cos(x)`` element-wise.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input data.

        Returns
        -------
        out: :ref:`IArray`
            The corresponding tangent values.

        References
        ----------
        `np.tan <https://numpy.org/doc/stable/reference/generated/numpy.tan.html>`_
        """
        return ia.LazyExpr(new_op=(self, "tan", None))

    def tanh(self):
        """
        Compute hyperbolic tangent element-wise.

        Equivalent to ``ia.sinh(x)/ia.cosh(x)``.

        Parameters
        ----------
        iarr: :ref:`IArray`
            Input data.

        Returns
        -------
        out: :ref:`IArray`
            The corresponding hyperbolic tangent values.

        References
        ----------
        `np.tanh <https://numpy.org/doc/stable/reference/generated/numpy.tanh.html>`_
        """
        return ia.LazyExpr(new_op=(self, "tanh", None))


@is_documented_by(IArray.abs)
def abs(iarr: IArray):
    return iarr.abs()


@is_documented_by(IArray.arccos)
def arccos(iarr: IArray):
    return iarr.arccos()


@is_documented_by(IArray.arcsin)
def arcsin(iarr: IArray):
    return iarr.arcsin()


@is_documented_by(IArray.arctan)
def arctan(iarr: IArray):
    return iarr.arctan()


@is_documented_by(IArray.arctan2)
def arctan2(iarr1: IArray, iarr2: IArray):
    return iarr1.arctan2(iarr2)


@is_documented_by(IArray.ceil)
def ceil(iarr: IArray):
    return iarr.ceil()


@is_documented_by(IArray.cos)
def cos(iarr: IArray):
    return iarr.cos()


@is_documented_by(IArray.cosh)
def cosh(iarr: IArray):
    return iarr.cosh()


@is_documented_by(IArray.exp)
def exp(iarr: IArray):
    return iarr.exp()


@is_documented_by(IArray.floor)
def floor(iarr: IArray):
    return iarr.floor()


@is_documented_by(IArray.log)
def log(iarr: IArray):
    return iarr.log()


@is_documented_by(IArray.log10)
def log10(iarr: IArray):
    return iarr.log10()


@is_documented_by(IArray.negative)
def negative(iarr: IArray):
    return iarr.negative()


@is_documented_by(IArray.power)
def power(iarr1: IArray, iarr2: IArray):
    return iarr1.power(iarr2)


@is_documented_by(IArray.sin)
def sin(iarr: IArray):
    return iarr.sin()


@is_documented_by(IArray.sinh)
def sinh(iarr: IArray):
    return iarr.sinh()


@is_documented_by(IArray.sqrt)
def sqrt(iarr: IArray):
    return iarr.sqrt()


@is_documented_by(IArray.tan)
def tan(iarr: IArray):
    return iarr.tan()


@is_documented_by(IArray.tanh)
def tanh(iarr: IArray):
    return iarr.tanh()


# Reductions


def reduce(
    a: IArray, method: ia.Reduce, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs
):
    if axis is None:
        axis = range(a.ndim)
    if isinstance(axis, int):
        axis = (axis,)

    shape = tuple([s for i, s in enumerate(a.shape) if i not in axis])

    if cfg is None:
        cfg = ia.get_config()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        c = ext.reduce_multi(cfg, a, method, axis)
        if c.ndim == 0:
            c = float(ia.iarray2numpy(c))
        return c


def max(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : :ref:`IArray`
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : :class:`Config` or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config` dataclass that should
        override the current configuration.
    Returns
    -------
    max : :ref:`IArray` or float
        Maximum of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """

    return reduce(a, ia.Reduce.MAX, axis, cfg, **kwargs)


def min(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : :ref:`IArray`
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : :class:`Config` or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config` dataclass that should
        override the current configuration.
    Returns
    -------
    min : :ref:`IArray` or float
        Minimum of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """
    return reduce(a, ia.Reduce.MIN, axis, cfg, **kwargs)


def sum(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Return the sum of array elements over a given axis.

    Parameters
    ----------
    a : :ref:`IArray`
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : :class:`Config` or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config` dataclass that should
        override the current configuration.
    Returns
    -------
    sum : :ref:`IArray` or float
        Sum of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """
    return reduce(a, ia.Reduce.SUM, axis, cfg, **kwargs)


def prod(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : :ref:`IArray`
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : :class:`Config` or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config` dataclass that should
        override the current configuration.
    Returns
    -------
    prod : :ref:`IArray` or float
        Product of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """
    return reduce(a, ia.Reduce.PROD, axis, cfg, **kwargs)


def mean(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    """
    Compute the arithmetic mean along the specified axis. Returns the average of the array elements.

    Parameters
    ----------
    a : :ref:`IArray`
        Input data.
    axis : None, int, tuple of ints, optional
        Axis or axes along which the reduction is performed. The default (axis = None) is perform
        the reduction over all dimensions of the input array.
        If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single
        axis or all the axes as default.
    cfg : :class:`Config` or None
        The configuration for this operation. If None (default), the current configuration will be
        used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config` dataclass that should
        override the current configuration.
    Returns
    -------
    mean : :ref:`IArray` or float
        Mean of a. If axis is None, the result is a float value. If axis is given, the result is
        an array of dimension a.ndim - len(axis).
    """
    return reduce(a, ia.Reduce.MEAN, axis, cfg, **kwargs)


# Linear Algebra


def opt_gemv(a: IArray, b: IArray, cfg=None, **kwargs):
    shape = (a.shape[0], b.shape[1]) if b.ndim == 2 else (a.shape[0],)

    if cfg is None:
        cfg = ia.get_config()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        return ext.opt_gemv(cfg, a, b)


def matmul_params(ashape, bshape, itemsize=8, l2_size=512 * 1024, chunk_size=128 * 1024 * 1024):
    """
    Given a matrix multiplication of two arrays, it computes the chunks and the blocks of the operands
    to use an optimized version of the matmul algorithm.

    Parameters
    ----------
    ashape: tuple or list
        The shape of the operand a.
    bshape: tuple or list
        The shape of the operand b.
    itemsize:
        The size of each item.
    l2_size: int
        The size of the l2 cache. It is used to compute the size of the blocks.
    chunk_size: int
        The maximum chunksize allowed. It is used to compute the size of the chunks.

    Returns
    -------
    params: tuple
        A tuple specifying the chunks and the blocks of the matmul operands a and b
        (achunks, ablocks, bchunks, bblocks).
    """
    if len(ashape) != 2:
        raise AttributeError("The dimension of a must be 2")
    if len(bshape) != 1 and len(bshape) != 2:
        raise AttributeError("The dimension of b must be 1 or 2")
    if ashape[1] != bshape[0]:
        raise AttributeError("ashape[1] must be equal to bshape[0]")

    if len(bshape) == 1:
        return matmul_gemv_params(ashape[0], ashape[1], itemsize, l2_size, chunk_size)
    else:
        return matmul_gemm_params(ashape[0], ashape[1], bshape[1], itemsize, l2_size, chunk_size)


def matmul_gemv_params(M, N, itemsize=8, l2_size=512 * 1024, chunk_size=128 * 1024 * 1024):
    """
    Given a matmul operation a * b = c, it computes the chunks and the blocks of the operands
    (a and b) to use an optimized version of the matmul algorithm.

    Parameters
    ----------
    M: int
        Specifies the number of rows of the matrix a and of the matrix c. M must be at least zero.
    N: int
        Specifies the number of columns of the matrix a and the number of rows of the vector b.
    itemsize:
        The size of each item.
    l2_size: int
        The size of the l2 cache. It is used to compute the size of the blocks.
    chunk_size: int
        The maximum chunksize allowed. It is used to compute the size of the chunks.

    Returns
    -------
    params: tuple
        A tuple specifying the chunks and the blocks of the matmul operands a and b
        (achunks, ablocks, bchunks, bblocks).
    """
    l2_nelem = l2_size // itemsize
    block_nelem_dim = int(-1 + np.sqrt(1 + l2_nelem))

    n_block = block_nelem_dim
    if n_block > N:
        n_block = N

    m_block = block_nelem_dim
    if m_block > M:
        m_block = M

    chunk_nelem = chunk_size // itemsize
    chunk_nelem_dim = int(np.sqrt(chunk_nelem))

    n_chunk = chunk_nelem_dim
    if n_chunk % n_block != 0:
        n_chunk = (n_chunk // n_block + 1) * n_block
    if n_chunk > N:
        if N % n_block == 0:
            n_chunk = N
        else:
            n_chunk = (N // n_block + 1) * n_block

    m_chunk = chunk_nelem_dim
    if m_chunk % m_block != 0:
        m_chunk = (m_chunk // m_block + 1) * m_block
    if m_chunk > M:
        if M % m_block == 0:
            m_chunk = M
        else:
            m_chunk = (M // m_block + 1) * m_block

    a_chunks = (m_chunk, n_chunk)
    a_blocks = (m_block, n_block)
    b_chunks = (n_chunk,)
    b_blocks = (n_block,)

    return a_chunks, a_blocks, b_chunks, b_blocks


def matmul_gemm_params(M, K, N, itemsize=8, l2_size=512 * 1024, chunk_size=128 * 1024 * 1024):
    """
    Given a matmul operation a * b = c, it computes the chunks and the blocks of the operands
    (a and b) to use an optimized version of the matmul algorithm.

    Parameters
    ----------
    M: int
        Specifies the number of rows of the matrix A and of the matrix C. M must be at least zero.
    K: int
        Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        K must be at least zero.
    N: int
        Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        N must be at least zero.
    itemsize: int
        The size of each item.
    l2_size: int
        The size of the l2 cache. It is used to compute the size of the blocks.
    chunk_size: int
        The maximum chunksize allowed. It is used to compute the size of the chunks.

    Returns
    -------
    params: tuple
        A tuple specifying the chunks and the blocks of the matmul operands a and b
        (achunks, ablocks, bchunks, bblocks).
    """
    l2_nelem = l2_size // itemsize
    block_nelem = l2_nelem // 3
    block_nelem_dim = int(np.sqrt(block_nelem))

    n_block = block_nelem_dim
    if n_block > N:
        n_block = N

    m_block = block_nelem_dim
    if m_block > M:
        m_block = M

    k_block = block_nelem_dim
    if k_block > K:
        k_block = K

    chunk_nelem = chunk_size // itemsize
    chunk_nelem_dim = int(np.sqrt(chunk_nelem))

    n_chunk = chunk_nelem_dim
    if n_chunk % n_block != 0:
        n_chunk = (n_chunk // n_block + 1) * n_block
    if n_chunk > N:
        if N % n_block == 0:
            n_chunk = N
        else:
            n_chunk = (N // n_block + 1) * n_block

    m_chunk = chunk_nelem_dim
    if m_chunk % m_block != 0:
        m_chunk = (m_chunk // m_block + 1) * m_block
    if m_chunk > M:
        if M % m_block == 0:
            m_chunk = M
        else:
            m_chunk = (M // m_block + 1) * m_block

    k_chunk = chunk_nelem_dim
    if k_chunk % k_block != 0:
        k_chunk = (k_chunk // k_block + 1) * k_block
    if k_chunk > K:
        if K % k_block == 0:
            k_chunk = K
        else:
            k_chunk = (K // k_block + 1) * k_block

    a_chunks = (m_chunk, k_chunk)
    b_chunks = (k_chunk, n_chunk)
    a_blocks = (m_block, k_block)
    b_blocks = (k_block, n_block)

    return a_chunks, a_blocks, b_chunks, b_blocks


def matmul(a: IArray, b: IArray, cfg=None, **kwargs):
    """Multiply two matrices.

    Parameters
    ----------
    a : :ref:`IArray`
        First array.
    b : :ref:`IArray`
        Second array.
    cfg : :class:`Config`
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :ref:`IArray`
        The resulting array.

    """
    shape = (a.shape[0], b.shape[1]) if b.ndim == 2 else (a.shape[0],)

    if cfg is None:
        cfg = ia.get_config()

    if (
        a.chunks
        and b.chunks
        and a.chunks[0] % a.blocks[0] == 0
        and a.chunks[1] % a.blocks[1] == 0
        and b.chunks[0] % b.blocks[0] == 0
        and a.chunks[1] == b.chunks[0]
        and a.blocks[1] == a.blocks[0]
        and "chunks" not in kwargs
        and "blocks" not in kwargs
    ):
        if b.ndim == 1:
            kwargs["chunks"] = (a.chunks[0],)
            kwargs["blocks"] = (a.blocks[0],)

            with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
                return ext.opt_gemv(cfg, a, b)

        elif b.ndim == 2 and b.chunks[1] % b.blocks[1] == 0:
            kwargs["chunks"] = (a.chunks[0], b.chunks[1])
            kwargs["blocks"] = (a.blocks[0], b.blocks[1])

            with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
                return ext.opt_gemm(cfg, a, b)

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        return ext.matmul(cfg, a, b)


def transpose(a: IArray, cfg=None, **kwargs):
    """Transpose an array.

    Parameters
    ----------
    a : :ref:`IArray`
        The array to transpose.
    cfg : :class:`Config`
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :ref:`IArray`
        The transposed array.

    """
    if a.ndim != 2:
        raise AttributeError("Array dimension must be 2")

    if cfg is None:
        cfg = ia.get_config()

    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.transpose(cfg, a)
