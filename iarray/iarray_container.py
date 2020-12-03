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


# For avoiding a warning in PyCharm in method signatures
IArray = None


class IArray:
    """The ironArray data container.

    This is not meant to be called from user space.
    """

    @property
    def data(self):
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
