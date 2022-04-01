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
import zarr
import s3fs

import iarray as ia
from iarray import iarray_ext as ext
from .utils import IllegalArgumentError, zarr_to_iarray_dtypes
from dataclasses import dataclass
from typing import Sequence


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


def empty(shape: Sequence, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return an empty array.

    An empty array has no data and needs to be filled via a write iterator.

    Parameters
    ----------
    shape : tuple, list
        The shape of the array to be created.
    cfg : :class:`Config`
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :ref:`IArray`
        The new array.
    """

    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.empty(cfg, dtshape)


def uninit(shape: Sequence, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return an uninitialized array.

    An uninitialized array has no data and needs to be filled via a write iterator.

    Parameters
    ----------
    shape : tuple, list
        The shape of the array to be created.
    cfg : :class:`Config`
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :ref:`IArray`
        The new array.
    """

    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.uninit(cfg, dtshape)


def arange(
    shape: Sequence, start=None, stop=None, step=None, cfg: ia.Config = None, **kwargs
) -> ia.IArray:
    """Return evenly spaced values within a given interval.

    `shape`, `cfg` and `kwargs` are the same than for :func:`empty`.

    `start`, `stop`, `step` are the same as in `np.arange <https://numpy.org/doc/stable/reference/generated/numpy.arange.html>`_.

    Returns
    -------
    :ref:`IArray`
        The new array.

    See Also
    --------
    empty : Create an empty array.
    """

    if (start, stop, step) == (None, None, None):
        stop = np.prod(shape)
        start = 0
        step = 1
    elif (stop, step) == (None, None):
        stop = start
        start = 0
        step = 1
    elif step is None:
        stop = stop
        start = start
        if shape is None:
            step = 1
        else:
            step = (stop - start) / np.prod(shape)
    slice_ = slice(start, stop, step)

    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.arange(cfg, slice_, dtshape)


def linspace(
    shape: Sequence, start: float, stop: float, cfg: ia.Config = None, **kwargs
) -> ia.IArray:

    """Return evenly spaced numbers over a specified interval.

    `shape`, `cfg` and `kwargs` are the same than for :func:`empty`.

    `start`, `stop` are the same as in `np.linspace <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_.

    Returns
    -------
    :ref:`IArray`
        The new array.

    See Also
    --------
    empty : Create an empty array.
    """
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.linspace(cfg, start, stop, dtshape)


def zeros(shape: Sequence, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a new array of given shape and type, filled with zeros.

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
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.zeros(cfg, dtshape)


def ones(shape: Sequence, cfg: ia.Config = None, **kwargs) -> ia.IArray:
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
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.ones(cfg, dtshape)


def full(shape: Sequence, fill_value: float, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a new array of given shape and type, filled with `fill_value`.

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
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(shape=shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(shape, cfg.dtype)
        return ext.full(cfg, fill_value, dtshape)


def zarr_proxy(zarr_urlpath, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a read-only Zarr proxy array.

    `cfg` and `kwargs` are the same than for :func:`empty` except by `nthreads`, which is
    always set to 1 (multi-threading is not yet supported).

    The data type and chunks must not differ from the original Zarr array.

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
    if ext._is_s3_store(zarr_urlpath):
        s3 = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=zarr_urlpath, s3=s3)
        z = zarr.open(store)
    else:
        z = zarr.open(zarr_urlpath)
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
        a = uninit(shape=z.shape, cfg=cfg)

    # Set special attr to identify zarr_proxy
    a.attrs["zproxy_urlpath"] = zarr_urlpath
    # Assign postfilter
    ext.set_zproxy_postfilter(a)
    return a
