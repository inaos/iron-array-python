###########################################################################################
# Copyright ironArray SL 2021.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of ironArray SL
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

import iarray as ia
from iarray import iarray_ext as ext
import numpy as np
from typing import Union
import ndindex
from .info import InfoReporter


class OIndex:
    def __init__(self, array):
        self.array = array

    def __getitem__(self, selection):
        return self.array.get_orthogonal_selection(selection)

    def __setitem__(self, selection, value):
        return self.array.set_orthogonal_selection(selection, value)


def process_selection(selection, shape):
    mask = tuple(True if isinstance(s, int) else False for s in selection)
    new_selection = []
    for s, n in zip(selection, shape):
        if isinstance(s, slice):
            si = np.array([i for i in range(*s.indices(n))])
        elif isinstance(s, int):
            si = np.array([s % n])
        else:
            si = np.array([i % n for i in s])
        new_selection.append(si)

    return new_selection


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

    @property
    def attrs(self):
        return ia.Attributes(self)

    @property
    def oindex(self):
        return OIndex(self)

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
            By default, this function deactives btune unless it is specified.

        Returns
        -------
        :ref:`IArray`
            The copy.
        """
        if cfg is None:
            cfg = self.cfg
            # the urlpath should not be copied
            cfg.urlpath = None

        # Generally we don't want btune to optimize, except if specified
        btune = False
        if "favor" in kwargs and "btune" not in kwargs:
            btune = True
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

    def resize(self, newshape, start=None):
        """Change the shape of the array by growing or shrinking one or more dimensions.

        Parameters
        ----------
        newshape : tuple or list
            The new shape of the array container. It should have the same dimensions
            as `self`.
        start: tuple, list or None, optional.
            The position from where the array will be extended or shrunk according to
            :paramref:`newshape`. If given, it should have the same dimensions
            as `self`. If None (the default), the appended or deleted chunks will happen
            at the end of the array.

        Notes
        -----
        The array values corresponding to the added positions are not initialized.
        Thus, the user is in charge of initializing them.
        Furthermore, the :paramref:`start` has to fulfill the same conditions than in
        :func:`insert`, :func:`append` and :func:`delete`.

        See Also
        --------
        insert
        append
        delete
        """
        ext.resize(self, new_shape=newshape, start=start)
        return self.shape

    def insert(self, data, axis=0, start=None):
        """Insert data in a position by extending the :paramref:`axis`.

        Parameters
        ----------
        data: object supporting the PyBuffer protocol
            The object containing the data.
        axis: int, optional
            The axis along the data contained by :paramref:`data` will be inserted.
            Default is 0.
        start: int, optional
            The position in the array axis from where to start inserting the data.
            If None (default), it will be appended at the end.

        Notes
        -----
        If :paramref:`start` is not at the end of the array, it must be a multiple of `chunks[axis]`.
        Furthermore, if `start != shape[axis]` the number of elements of :paramref:`data`
        must be a multiple of `chunks[axis] * shape[in the other axis]` and
        if `start = shape[axis]` (or `None`) the number of elements of :paramref:`data`
        must be a multiple of `shape[in the other axis]`.

        For example, let’s suppose that we have an array of `shape = [20, 20]` and `chunks = [7,7]`,
        and we would like to insert data in the `axis = 0`. Then, if `start = 0`
        which is different from `shape[axis]` and multiple of `chunks[axis]`,
        the number of elements of :paramref:`data` must be a multiple of `7 * 20`.
        If `start = 20 = shape[axis]` (or None), the number of elements of :paramref:`data`
        can be `anything * 20`.

        See Also
        --------
        append
        delete
        resize
        """
        if type(data) is np.ndarray:
            if data.dtype.itemsize != np.dtype(self.dtype).itemsize:
                data = np.array(data, dtype=self.dtype)
            elif data.dtype.str[0] == ">":
                data = data.byteswap()
        ext.insert(self, data, axis, start)
        return self.shape

    def append(self, data, axis=0):
        """Append data at the end by extending the :paramref:`axis`.

        Parameters
        ----------
        data: object supporting the PyBuffer protocol
            The object containing the data.
        axis: int, optional
            Axis along which to append.
            Default is 0.

        Notes
        -----
        The number of elements of :paramref:`data` must be a multiple of the array shape in all its axis
        excluding the :paramref:`axis`.

        For example, let’s suppose that we have an array of `shape = [20, 20]`, and `chunks = [7, 7]`.
        Then number of elements of :paramref:`data` can be `anything * 20` and the new shape would be
        `[20 + anything, 20]`.

        See Also
        --------
        insert
        delete
        resize
        """
        if type(data) is np.ndarray:
            if data.dtype.itemsize != np.dtype(self.dtype).itemsize:
                data = np.array(data, dtype=self.dtype)
            elif data.dtype.str[0] == ">":
                data = data.byteswap()
        ext.append(self, data, axis)
        return self.shape

    def delete(self, delete_len, axis=0, start=None):
        """Delete :paramref:`delete_len` positions along the :paramref:`axis` from the
        :paramref:`start`.

        Parameters
        ----------
        delete_len: int
            The number of elements to delete in the `array.shape[axis]`.
        axis: int, optional
            The axis that will be shrunk.
            Default is 0.
        start: int, None, optional
            The starting point for deleting the elements. If None (default)
            the deleted elements will be at the end of the array.

        Notes
        -----
        If :paramref:`delete_len` is not a multiple of `chunks[axis]`,
        :paramref:`start` must be either None or `shape[axis] - delete_len` (which are equivalent).
        Otherwise, :paramref:`start` must also be a multiple of `chunks[axis]`.

        For example, let’s suppose that we have an array with `shape = [20, 20]` and `chunks = [7, 7]`.
        If `delete_len = 5` and `axis = 0`, because :paramref:`delete_len`
        is not a multiple of `chunks[axis]`, :paramref:`start`
        must be `None` or `shape[axis] - delete_len = 15`. In both cases, the deleted elements
        will be the same (those at the end) and the new shape will be `[15, 20]`.
        If we would like to delete some elements
        in the middle of the array, :paramref:`start` and :paramref:`delete_len` both must be a multiple
        of `chunks[axis]`. So the only possibilities is this particular case would be
        `start = 0` and `delete_len = 7` or `delete_len = 14` which would give an array with
        shape `[13, 20]` or `[6, 20]`. Or `start = 7` and
        `delete_len = 7` which would give an array with shape `[13, 20]`.

        See Also
        --------
        resize
        insert
        append
        """
        ext.delete(self, axis=axis, start=start, delete_len=delete_len)
        return self.shape

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
        if isinstance(key, ia.LazyExpr):
            return key.update_expr(new_op=(self, f"[]", key))
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
            if self.np_dtype is None:
                value = value.data
            else:
                value = value.data
                value = value.astype(dtype=self.dtype)
        elif self.np_dtype is not None:
            value = np.full(shape, value, dtype=self.dtype)
        with ia.config(cfg=self.cfg) as cfg:
            return ext.set_slice(cfg, self, start, stop, value)

    def set_orthogonal_selection(self, selection, value):
        """Modify data via a selection for each dimension of the array.

        Parameters
        ----------
        selection: int, slice or integer array.
            The selection for each dimension of the array.
        value: value or `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_.
            Value to be stored into the array.

        Returns
        -------
        None

        Notes
        -----
        This function can also be replaced by `self.oindex[selection]`.

        See Also
        --------
        get_orthogonal_selection
        """
        selection = process_selection(selection, self.shape)
        if type(value) == list:
            value = np.array(value)
        elif isinstance(value, (int, float)):
            shape = [len(s) for s in selection]
            value = np.full(shape, value, self.dtype)
        with ia.config(cfg=self.cfg) as cfg:
            return ext.set_orthogonal_selection(cfg, self, selection, value)

    def get_orthogonal_selection(self, selection):
        """Retrieve data by making a selection for each dimension of the array.

        Parameters
        ----------
        selection: list
            The selection for each dimension. It can be either
            an integer (indexing a single item), a slice or an array of integers.

        Returns
        -------
        out: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_

        Notes
        -----
        This function can also be replaced by `self.oindex[selection]`.

        See Also
        --------
        set_orthogonal_selection
        """
        selection = process_selection(selection, self.shape)
        shape = tuple(len(s) for s in selection)
        with ia.config(cfg=self.cfg) as cfg:
            dst = np.ones(shape, dtype=self.dtype)
            return ext.get_orthogonal_selection(cfg, self, dst, selection)

    def astype(self, view_dtype):
        """
        Cast the array into a view of a specified type.

        Parameters
        ----------
        view_dtype: (np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16,
            np.uint8, np.bool_)
            The dtype in which the array will be casted. Only upcasting is supported.

        Returns
        -------
        :ref:`IArray`
            The new view as a normal IArray.
        """
        view_dtypesize = np.dtype(view_dtype).itemsize
        src_dtypesize = np.dtype(self.dtype).itemsize
        if view_dtypesize < src_dtypesize:
            raise OverflowError("`view_dtype` itemsize must be greater or equal than `self.dtype`")
        return ext.get_type_view(self.cfg, self, view_dtype)

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

    def __lt__(self, value):
        return ia.LazyExpr(new_op=(self, "<", value))

    def __le__(self, value):
        return ia.LazyExpr(new_op=(self, "<=", value))

    def __gt__(self, value):
        return ia.LazyExpr(new_op=(self, ">", value))

    def __ge__(self, value):
        return ia.LazyExpr(new_op=(self, ">=", value))

    def __eq__(self, value):
        if ia._disable_overloaded_equal:
            return self is value
        return ia.LazyExpr(new_op=(self, "==", value))

    def __ne__(self, value):
        return ia.LazyExpr(new_op=(self, "!=", value))

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
        iarr(self): :ref:`IArray`
           Input array.

        Returns
        -------
        abs: :ref:`iarray.Expr`
           A lazy expression that must be evaluated via `out.eval()`, which will compute the
           absolute value of each element in x.

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
        iarr(self): :ref:`IArray`
            x-coordinate on the unit circle. For real arguments, the domain is :math:`\\left [ -1, 1 \\right]`.

        Returns
        -------
        angle: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the
            angle of the ray intersecting the unit circle at the given x-coordinate in radians
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
        iarr(self): :ref:`IArray`
           y-coordinate on the unit circle.

        Returns
        -------
        angle: :ref:`iarray.Expr`
           A lazy expression that must be evaluated via `out.eval()`, which will compute the inverse
           sine of each element in :math:`x`, in radians and in the closed interval
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
        iarr(self): :ref:`IArray`
            Input array.

        Returns
        -------
        angle: :ref:`iarray.Expr`
           A lazy expression that must be evaluated via `out.eval()`, which will compute the
           angles in radians, in the range :math:`\\left[-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right]`.

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

    def arctan2(self, iarr2):
        """
        Element-wise arc tangent of :math:`\\frac{iarr_1}{iarr_2}` choosing the quadrant correctly.


        Parameters
        ----------
        iarr1(self): :ref:`IArray`
            y-coordinates.
        iarr2: :ref:`IArray`
            x-coordinates.

        Returns
        -------
        angle: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the
            angles in radians, in the range :math:`[-\\pi, \\pi]`.

        References
        ----------
        `np.arctan2 <https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html>`_
        """
        return ia.LazyExpr(new_op=(self, "atan2", iarr2))

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
        iarr(self): :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the
            ceiling of each element in :math:`x`.

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
        iarr(self): :ref:`IArray`
            Angle, in radians.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the actual
            cosine values.

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
        iarr(self): :ref:`IArray`
            Input data.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the actual
            hyperbolic cosine values.

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
        iarr(self): :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the
            element-wise exponential of input data.

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
        iarr(self): :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the
            floor of each element in input data.

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
        iarr(self): :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the
            natural logarithm of input data, element-wise.

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
        iarr(self): :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the
            logarithm to the base 10 of input data, element-wise.

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
        iarr(self): :ref:`IArray`
            Input array.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute
            :math:`out = -iarr`.

        References
        ----------
        `np.negative <https://numpy.org/doc/stable/reference/generated/numpy.negative.html>`_
        """
        return ia.LazyExpr(new_op=(self, "negate", None))

    def power(self, iarr2):
        """
        First array elements raised to powers from second array, element-wise.

        Parameters
        ----------
        iarr1(self): :ref:`IArray`
            The bases.
        iarr2: :ref:`IArray`
            The exponents.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the
            bases raised to the exponents.

        References
        ----------
        `np.power <https://numpy.org/doc/stable/reference/generated/numpy.power.html>`_
        """
        return ia.LazyExpr(new_op=(self, "pow", iarr2))

    def sin(self):
        """
        Trigonometric sine, element-wise.

        Parameters
        ----------
        iarr(self): :ref:`IArray`
            Angle, in radians.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the actual
            sine values.

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
        iarr(self): :ref:`IArray`
            Input data.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the actual
            hyperbolic sine values.

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
        iarr(self): :ref:`IArray`
            The values whose square-roots are required.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the actual
            positive square-root of each element in input data.

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
        iarr(self): :ref:`IArray`
            Input data.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the actual
            tangent values.

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
        iarr(self): :ref:`IArray`
            Input data.

        Returns
        -------
        out: :ref:`iarray.Expr`
            A lazy expression that must be evaluated via `out.eval()`, which will compute the actual
            hyperbolic tangent values.

        References
        ----------
        `np.tanh <https://numpy.org/doc/stable/reference/generated/numpy.tanh.html>`_
        """
        return ia.LazyExpr(new_op=(self, "tanh", None))

    def min(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """
        Return the minimum of an array or minimum along an axis.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Minimum of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.MIN, axis, cfg, **kwargs)

    def max(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """
        Return the maximum of an array or maximum along an axis.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Maximum of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.MAX, axis, cfg, **kwargs)

    def sum(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """
        Return the sum of array elements over a given axis.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Sum of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). Its `dtype` is `np.int64` for integers and bools,
            `np.uint64` for unsigned integers and the `dtype` of :paramref:`a` otherwise.
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=self.cfg) as cfg:
            return reduce(self, ia.Reduce.SUM, axis, cfg, **kwargs)

    def prod(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """
        Return the product of array elements over a given axis.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Product of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). Its `dtype` is `np.int64` for integers and bools,
            `np.uint64` for unsigned integers and the `dtype` of :paramref:`a` otherwise.
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.PROD, axis, cfg, **kwargs)

    def mean(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """
        Compute the arithmetic mean along the specified axis. Returns the average of the array elements.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Mean of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). Its `dtype` is `np.float32` when the `dtype` of
            :paramref:`a` is `np.float32` and `np.float64` otherwise.
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.MEAN, axis, cfg, **kwargs)

    def std(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """Returns the standard deviation, a measure of the spread of a distribution,
        of the array elements. The standard deviation is computed for the flattened
        array by default, otherwise over the specified axis.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
        std : :ref:`IArray` or float
            Standard deviation of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). Its `dtype` is `np.float32` when the `dtype` of
            :paramref:`a` is `np.float32` and `np.float64` otherwise.
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.STD, axis, cfg, **kwargs)

    def var(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """Compute the variance along the specified axis. Returns the variance of the array elements,
        a measure of the spread of a distribution. The variance is computed for the flattened
        array by default, otherwise over the specified axis.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
        var : :ref:`IArray` or float
            Variance of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). Its `dtype` is `np.float32` when the `dtype` of
            :paramref:`a` is `np.float32` and `np.float64` otherwise.
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.VAR, axis, cfg, **kwargs)

    def median(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """
        Compute the median along the specified axis. Returns the median of the array elements.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
        median : :ref:`IArray` or float
            Median of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). Its `dtype` is `np.float32` when the `dtype` of
            :paramref:`a` is `np.float32` and `np.float64` otherwise.
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.MEDIAN, axis, cfg, **kwargs)

    def nanmin(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """
        Return the minimum of an array or minimum along an axis ignoring NaNs.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Minimum of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.

        See Also
        --------
        min
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.NAN_MIN, axis, cfg, **kwargs)

    def nanmax(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """
        Return the maximum of an array or maximum along an axis ignoring NaNs.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Maximum of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.

        See Also
        --------
        max
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.NAN_MAX, axis, cfg, **kwargs)

    def nansum(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """
        Return the sum of array elements over a given axis ignoring NaNs.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Sum of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.

        See Also
        --------
        sum
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.NAN_SUM, axis, cfg, **kwargs)

    def nanprod(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """Return the product of array elements over a given axis ignoring NaNs.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Product of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.

        See Also
        --------
        prod
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.NAN_PROD, axis, cfg, **kwargs)

    def nanmean(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """Compute the arithmetic mean along the specified axis ignoring NaNs.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
            Mean of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.

        See Also
        --------
        mean
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.NAN_MEAN, axis, cfg, **kwargs)

    def nanstd(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """Returns the standard deviation  ignoring NaNs.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
        std : :ref:`IArray` or float
            Standard deviation of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.

        See Also
        --------
        std
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.NAN_STD, axis, cfg, **kwargs)

    def nanvar(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """Compute the variance along the specified axis ignoring NaNs. The variance is computed for the flattened
        array by default, otherwise over the specified axis.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
        var : :ref:`IArray` or float
            Variance of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.

        See Also
        --------
        var
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.NAN_VAR, axis, cfg, **kwargs)

    def nanmedian(self, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
        """Compute the median ignoring NaNs along the specified axis. Returns the median of the array elements.

        Parameters
        ----------
        a(self) : :ref:`IArray`
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
        median : :ref:`IArray` or float
            Median of a. If axis is None, the result is a value. If axis is given, the result is
            an array of dimension a.ndim - len(axis). The `dtype` is always the `dtype` of :paramref:`a`.

        See Also
        --------
        median
        """
        cfg = self.cfg if cfg is None else cfg
        with ia.config(cfg=cfg) as cfg:
            return reduce(self, ia.Reduce.NAN_MEDIAN, axis, cfg, **kwargs)

    @attrs.setter
    def attrs(self, value):
        self._attrs = value


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
        cfg = ia.get_config_defaults()

    dtype = kwargs.get("dtype")
    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        c = ext.reduce_multi(cfg, a, method, axis)
        if dtype is not None and dtype != c.dtype:
            raise RuntimeError("Cannot set the result's data type")
        if c.ndim == 0:
            c = c.dtype(ia.iarray2numpy(c))
        return c


@is_documented_by(IArray.max)
def max(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.max(axis, cfg, **kwargs)


@is_documented_by(IArray.min)
def min(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.min(axis, cfg, **kwargs)


@is_documented_by(IArray.sum)
def sum(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.sum(axis, cfg, **kwargs)


@is_documented_by(IArray.prod)
def prod(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.prod(axis, cfg, **kwargs)


@is_documented_by(IArray.mean)
def mean(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.mean(axis, cfg, **kwargs)


@is_documented_by(IArray.std)
def std(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.std(axis, cfg, **kwargs)


@is_documented_by(IArray.var)
def var(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.var(axis, cfg, **kwargs)


@is_documented_by(IArray.median)
def median(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.median(axis, cfg, **kwargs)


@is_documented_by(IArray.nanmax)
def nanmax(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.nanmax(axis, cfg, **kwargs)


@is_documented_by(IArray.nanmin)
def nanmin(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.nanmin(axis, cfg, **kwargs)


@is_documented_by(IArray.nansum)
def nansum(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.nansum(axis, cfg, **kwargs)


@is_documented_by(IArray.nanprod)
def nanprod(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.nanprod(axis, cfg, **kwargs)


@is_documented_by(IArray.nanmean)
def nanmean(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.nanmean(axis, cfg, **kwargs)


@is_documented_by(IArray.nanstd)
def nanstd(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.nanstd(axis, cfg, **kwargs)


@is_documented_by(IArray.nanvar)
def nanvar(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.nanvar(axis, cfg, **kwargs)


@is_documented_by(IArray.nanmedian)
def nanmedian(a: IArray, axis: Union[int, tuple] = None, cfg: ia.Config = None, **kwargs):
    return a.nanmedian(axis, cfg, **kwargs)


# Linear Algebra


def opt_gemv(a: IArray, b: IArray, cfg=None, **kwargs):
    shape = (a.shape[0], b.shape[1]) if b.ndim == 2 else (a.shape[0],)

    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        return ext.opt_gemv(cfg, a, b)


def matmul_params(ashape, bshape, dtype=None, l2_size=None, chunk_size=128 * 1024 * 1024):
    """
    Given a matrix multiplication of two arrays, it computes the chunks and the blocks of the operands
    to use an optimized version of the matmul algorithm.

    Parameters
    ----------
    ashape: tuple or list
        The shape of the operand a.
    bshape: tuple or list
        The shape of the operand b.
    dtype:
        The dtype of each item.
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

    if not dtype:
        dtype = ia.get_config_defaults().dtype
    itemsize = np.dtype(dtype).itemsize

    if not l2_size:
        l2_size = ia.get_l2_size()

    l2_size = l2_size // 2
    # The above operation is based on the following results:
    # Matmul performance on Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz (14 cores, 28 logical)
    # Time (L2 size = 65536) 4.27 s
    # Time (L2 size = 131072) 2.53 s
    # Time (L2 size = 262144) 1.81 s
    # Time (L2 size = 524288) 1.89 s
    # Time (L2 size = 1048576) 4.68 s  <- CPU L2 size
    # Time (L2 size = 2097152) 4.06 s

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
        cfg = ia.get_config_defaults()

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
        If None (default), global defaults are used. The `np_dtype` will be the one
        from  :paramref:`a`.
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
        cfg = ia.get_config_defaults()

    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.transpose(cfg, a)
