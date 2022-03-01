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
import zarr

import iarray as ia
from iarray import iarray_ext as ext
from dataclasses import dataclass, field
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
    dtype: (np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16,
        np.uint8, np.bool_) = np.float64

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


zarr_to_iarray_dtypes = {'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
                         'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
                         'float32': np.float32, 'float64': np.float64, 'bool': np.bool_}

def zarr_proxy(zarr_urlpath, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a zarr proxy array.


    `cfg` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    :ref:`IArray`
        The zarr proxy array.

    See Also
    --------
    empty : Create an empty array.
    """
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
            if tuple(kwargs.pop("blocks")) != z.chunks:
                raise AttributeError("blocks cannot differ from chunks")

    with ia.config(cfg=cfg, dtype=dtype, chunks=z.chunks, blocks=z.chunks, **kwargs) as cfg:
        a = uninit(shape=z.shape, cfg=cfg)

    # Set special vlmeta to identify zarr_proxy
    a.vlmeta["zproxy_urlpath"] = zarr_urlpath
    # Assign postfilter
    ext.set_zproxy_postfilter(a)
    return a
