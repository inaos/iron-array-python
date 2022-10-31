###########################################################################################
# Copyright ironArray SL 2021.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of ironArray SL
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

import numpy as np

import iarray as ia
from iarray import iarray_ext as ext
from .utils import IllegalArgumentError, zarr_to_iarray_dtypes
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from .dtypes import (
    _all_dtypes,
    _boolean_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _floating_dtypes,
    _numeric_dtypes,
    _dtype_categories,
)


_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /): ...

    def __len__(self, /): ...


SupportsBufferProtocol = Any


@dataclass
class DTShape:
    """Shape and data type dataclass.

    Parameters
    ----------
    shape: list, tuple
        The shape of the array.
    dtype: (np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16,
        np.uint8, np.bool_)
        The data type of the elements in the array.  The default is np.float64.
    """

    shape: Sequence
    dtype: (
        np.float64,
        np.float32,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
        np.bool_,
    ) = np.float64

    def __post_init__(self):
        if self.shape is None:
            raise ValueError("shape must be non-empty")


def empty(shape: Union[int, Tuple[int, ...]],
          *,
          device: Optional[ia.Device] = None,
          cfg: ia.Config = None,
          **kwargs
          ) -> ia.IArray:
    """Return an uninitialized array.

    An empty array has no data and needs to be filled via a write iterator.

    Parameters
    ----------
    shape : int, tuple
        The shape of the array to be created.
    device: Device
        The device on which to place the created array. The only supported value is `"cpu"`.
    cfg : :class:`Config`
        The configuration to use.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :ref:`IArray`
        The new array.
    """
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.uninit(cfg, dtshape)


def empty_like(iarr: ia.IArray, /, *, device: Optional[ia.Device] = None,
               cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Returns an uninitialized array with the same shape as an input array :paramref:`iarr`.

    Parameters
    ----------
    iarr: :ref:`IArray`
    device: Device
        The device on which to place the created array. The only supported value is `"cpu"`.
    cfg: :class:`Config`
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs: dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :ref:`IArray`
        The new array.
    """
    return empty(iarr.shape, device=device, cfg=cfg, **kwargs)


def arange(start: Union[int, float],
           /,
           stop: Optional[Union[int, float]] = None,
           step: Union[int, float] = 1,
           *,
           shape: Sequence = None,
           device: Optional[ia.Device] = None,
           cfg: ia.Config = None,
           **kwargs
           ) -> ia.IArray:
    """Return evenly spaced values within a given interval.

    `shape`, `device`, `cfg` and `kwargs` are the same than for :func:`empty`.

    `start`, `stop`, `step` are the same as in `np.arange <https://numpy.org/doc/stable/reference/generated/numpy.arange.html>`_.

    Returns
    -------
    :ref:`IArray`
        The new array.

    See Also
    --------
    empty : Create an empty array.
    """
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        if (
            cfg.np_dtype is not None
            and type(start) in [np.datetime64, np.timedelta64]
            and start.dtype.str[1:] != cfg.np_dtype[1:]
        ):
            raise ValueError("`start` has to be the same type as `cfg.np_dtype`")
        if (
            cfg.np_dtype is not None
            and type(stop) in [np.datetime64, np.timedelta64]
            and stop.dtype.str[1:] != cfg.np_dtype[1:]
        ):
            raise ValueError("`stop` has to be the same type as `cfg.np_dtype`")

        if step == 0:
            raise ValueError("`step` cannot be 0")
        if stop is None:
            if cfg.np_dtype is not None:
                # For datetimes
                stop = np.array(start, dtype=cfg.dtype)
            else:
                stop = start
            start = np.array(0, dtype=cfg.dtype)
        elif cfg.np_dtype is not None:
            # For datetimes
            stop = np.array(stop, dtype=cfg.dtype)
            start = np.array(start, dtype=cfg.dtype)

        if (stop - start <= 0 and step > 0) or (stop - start >= 0 and step < 0):
            # Return 0 length array
            shape = (0,)
            return empty(shape, cfg=cfg, **kwargs)
        if shape is None:
            shape = [np.ceil((stop - start) / step)]

        slice_ = slice(start, stop, step)
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.arange(cfg, slice_, dtshape)


def asarray(obj: Union[ia.IArray, bool, int, float, NestedSequence, SupportsBufferProtocol],
            /, *, device: Optional[ia.Device] = None,
            copy: Optional[bool] = None, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """
    Convert the input to an :ref:`IArray`.

    Parameters
    ----------
    obj: :ref:`IArray`, Python scalar, (possibly nested) sequence of Python scalars, or object supporting the Python buffer protocol
        The input to convert into an :ref:`IArray`.
    device: Device
        The device on which to place the created array. The only supported value is `"cpu"`.
    copy: bool
        Whether to copy the buffer data in case of an :ref:`IArray` instance.
    cfg: :class:`ia.Config`
        The configuration to use.
        If None (default), global defaults are used.
    kwargs: dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    out: :ref:`IArray`
        An array containing the data of :paramref:`obj`.
    """
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if cfg is None:
        cfg = ia.get_config_defaults()
    if isinstance(obj, ia.IArray):
        with ia.config(cfg=cfg, **kwargs) as cfg:
            if copy is None:
                if np.dtype(cfg.dtype).itemsize < np.dtype(obj.dtype).itemsize:
                    copy = True
                else:
                    copy = False
            if copy:
                return obj.copy(cfg=cfg)
            else:
                if cfg.urlpath is not None or cfg.contiguous not in [None, obj.cfg.contiguous]\
                        or cfg.chunks not in [None, obj.chunks] \
                        or cfg.blocks not in [None, obj.blocks]:
                    raise ValueError("Cannot change array config when avoiding the copy")

                if cfg.dtype == obj.dtype:
                    return obj[...]
                else:
                    return ia.astype(obj, cfg.dtype)
    else:
        copy = True if copy is None else copy
        if not copy:
            raise ValueError("Cannot avoid copy for non IArray instances")
        with ia.config(cfg=cfg, **kwargs) as cfg:
            dtype = cfg.dtype if cfg.np_dtype is None else cfg.np_dtype
            arr = np.asarray(obj, dtype=dtype)
            res = ia.empty(arr.shape)
            if arr.ndim == 0:
                res[()] = arr[()]
            else:
                res[...] = arr[...]
            del arr
            return res


def linspace(start: Union[int, float], stop: Union[int, float], /, num: int, *,
             shape: Sequence = None, device: Optional[
            ia.Device] = None, endpoint: bool = True, cfg: ia.Config = None, **kwargs
             ) -> ia.IArray:
    """Return evenly spaced numbers over a specified interval. If :paramref:`endpoint` is False,
    the numbers will be generated over the half-open interval `[start, stop)`.

    `shape`, `device`, `cfg` and `kwargs` are the same than for :func:`empty`.

    `start`, `stop` are the same as in `np.linspace <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_.

    Returns
    -------
    :ref:`IArray`
        The new array.

    See Also
    --------
    empty : Create an empty array.
    """
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    if shape is None:
        shape = [num]
    elif np.prod(shape) != num:
        raise ValueError("`shape` must agree with `num`")
    if not endpoint and len(shape) > 1:
        raise ValueError("`endpoint` can only be False with 1-dim arrays")

    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        if not endpoint:
            shape = [num + 1]
        dtshape = ia.DTShape(shape, cfg.dtype)
        a = ext.linspace(cfg, start, stop, dtshape)
        if not endpoint:
            a.resize([num])
        return a


def zeros(shape: Union[int, Tuple[int, ...]], *, device: Optional[ia.Device] = None, cfg: ia.Config = None,
          **kwargs) -> ia.IArray:
    """Return a new array of given shape and type, filled with zeros.

    `shape`, `device`, `cfg` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    :ref:`IArray`
        The new array.

    See Also
    --------
    empty : Create an empty array.
    ones : Create an array filled with ones.
    """
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        if cfg.dtype not in _all_dtypes:
            raise TypeError("dtype is not supported")
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.zeros(cfg, dtshape)


def zeros_like(iarr: ia.IArray, /, *, device: Optional[ia.Device] = None,
               cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a new array of same shape as :paramref:`iarr`, filled with zeros.

    `shape`, `cfg` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    :ref:`IArray`
        The new array.

    See Also
    --------
    empty : Create an empty array.
    ones : Create an array filled with ones.
    """
    return zeros(iarr.shape, device=device, cfg=cfg, **kwargs)


def concatenate(shape: Sequence, data: list, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Concatenate a list of one-chunk arrays into a specified shape.

    `cfg` and `kwargs` are the same as for :func:`empty`.

    Parameters
    ----------
    shape: Sequence
        The shape of the concatenated array.
    data: list
        A list with the arrays (with one chunk) to concatenate

    Returns
    -------
    :ref:`IArray`
        The concatenated array.
    """
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(
        shape=shape, cfg=cfg, chunks=data[0].chunks, blocks=data[0].blocks, **kwargs
    ) as cfg:
        dtshape = ia.DTShape(shape, data[0].dtype)
        return ext.concatenate(cfg, data, dtshape)


def from_cframe(
    cframe: [bytes, bytearray], copy: bool = False, cfg: ia.Config = None, **kwargs
) -> ia.IArray:
    """
    Create an array from a cframe in bytes.

    Parameters
    ----------
    cframe: bytes
        The cframe in bytes.
    copy: bool
        If `copy` is True, a copy is made.

    cfg` and `kwargs` are the same as for :func:`empty`.

    Returns
    -------
    :ref:`IArray`
        The new array.
    """
    if not cfg:
        cfg = ia.get_config_defaults()

    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.from_cframe(cfg, cframe, copy)


def ones(shape: Union[int, Tuple[int, ...]], *, device: Optional[ia.Device] = None, cfg: ia.Config = None,
         **kwargs) -> ia.IArray:
    """Return a new array of given shape and type, filled with ones.

    `shape`, `cfg` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    :ref:`IArray`
        The new array.

    See Also
    --------
    empty : Create an empty array.
    zeros : Create an array filled with zeros.
    """
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        if cfg.dtype not in _all_dtypes:
            raise TypeError("dtype is not supported")
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.ones(cfg, dtshape)


def ones_like(iarr: ia.IArray, /, *, device: Optional[ia.Device] = None,
              cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a new array with the same shape as an input array :paramref:`iarr`, filled with ones.

    Parameters
    ----------
    iarr: :ref:`IArray`
    device: ia.Device
        The device on which to place the created array. The only supported value is `"cpu"`.
    cfg: :class:`Config`
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs: dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :ref:`IArray`
        The new array.
    """
    return ones(iarr.shape, device=device, cfg=cfg, **kwargs)


def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[bool, int, float], *, device: Optional[ia.Device] = None,
         cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a new array of given shape and type, filled with :paramref:`fill_value`.

    `shape`, `device`, `cfg` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    :ref:`IArray`
        The new array.

    See Also
    --------
    empty : Create an empty array.
    zeros : Create an array filled with zeros.
    """
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        if cfg.dtype not in _all_dtypes:
            raise TypeError("dtype is not supported")
        dtshape = ia.DTShape(shape, cfg.dtype)
        if (
            cfg.np_dtype is not None
            and type(fill_value) in [np.datetime64, np.timedelta64]
            and fill_value.dtype.str[1:] != cfg.np_dtype[1:]
        ):
            raise ValueError("`fill_value` has to be the same type as `cfg.np_dtype`")
        fill_value = np.array(fill_value, dtype=cfg.dtype)
        return ext.full(cfg, fill_value, dtshape)


def full_like(iarr: ia.IArray, /, fill_value: Union[bool, int, float], *, device: Optional[ia.Device] = None,
              cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a new array with the same shape as an input array :paramref:`iarr`, filled with :paramref:`fill_value`.

    `fill_value`, `device`, `cfg` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    :ref:`IArray`
        The new array.
    """
    return full(iarr.shape, fill_value=fill_value, device=device, cfg=cfg, **kwargs)


def zarr_proxy(zarr_urlpath, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a read-only Zarr proxy array.

    `cfg` and `kwargs` are the same than for :func:`empty` except by `nthreads`, which is
    always set to 1 (multi-threading is not yet supported).

    The data type and chunks must not differ from the original Zarr array.

    A Zarr proxy is a regular IArray array but with a special attribute called `zproxy_urlpath`. This
    attribute is protected when `attrs.clear()` is used; but can still be deleted with `del attrs["zproxy_urlpath"]`,
    `attrs.popitem()` or `attrs.pop("zproxy_urlpath")`.

    This IArray has an additional attribute called `proxy_attrs` which contains the Zarr attributes. The user can
    get and set these attributes.

    Parameters
    ----------
    zarr_urlpath : str
        The path to the Zarr array.
        If it is stored in the cloud, the path must begin with ``s3://``.

    Returns
    -------
    :ref:`IArray`
        The zarr proxy array.

    Notes
    -----
    As a proxy, this array does not contain the data from the original array, it only reads it when needed.
    But if a :func:`save` is done, a copy of all the data will be made and assigned to a new
    and usual on disk :ref:`IArray`. To create a persistent proxy on-disk,
    you can specificy the :paramref:`urlpath` during :func:`zarr_proxy` execution time.
    """
    z = ext._zarray_from_proxy(zarr_urlpath)
    # Create iarray
    dtype = zarr_to_iarray_dtypes[str(z.dtype)]

    if cfg is None:
        cfg = ia.get_config_defaults()

    if kwargs != {}:
        if "dtype" in kwargs:
            if kwargs.pop("dtype") != dtype:
                raise AttributeError("dtype cannot differ from the original array")
        if "chunks" in kwargs:
            if tuple(kwargs.pop("chunks")) != z.chunks:
                raise AttributeError("chunks cannot differ from the original array")
        if "blocks" in kwargs:
            blocks = tuple(kwargs.pop("blocks"))
        else:
            blocks = z.chunks
        if "nthreads" in kwargs:
            if kwargs.pop("nthreads") != 1:
                raise IllegalArgumentError("Cannot use parallelism when interacting with Zarr")

    with ia.config(
        cfg=cfg, dtype=dtype, chunks=z.chunks, blocks=blocks, nthreads=1, **kwargs
    ) as cfg:
        a = empty(shape=z.shape, cfg=cfg)

    # Set special attr to identify zarr_proxy
    a.attrs["zproxy_urlpath"] = zarr_urlpath
    # Create reference to zarr.attrs
    a.zarr_attrs = z.attrs
    # Assign postfilter
    ext.set_zproxy_postfilter(a)
    return a
