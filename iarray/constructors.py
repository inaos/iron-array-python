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
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
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
