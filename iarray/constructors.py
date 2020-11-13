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
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DTShape:
    shape: Sequence
    dtype: (np.float32, np.float64) = np.float64
    """Shape and data type dataclass.

    Parameters
    ----------
    shape: list, tuple
        The shape of the array.
    dtype: np.float32, np.float64
        The data type of the elements in the array.  The default is np.float64.
    """

    def __post_init__(self):
        if not self.shape:
            raise ValueError("shape must be non-empty")


def empty(dtshape: DTShape, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return an empty array.

    An empty array has no data and needs to be filled via a write iterator.

    Parameters
    ----------
    dtshape : DTShape
        The shape and data type of the array to be created.
    cfg : Config
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config
        dataclass that should override the current configuration.

    Returns
    -------
    IArray
        The new array.

    See Also
    --------
    IArray.iter_write_block : Iterator for filling an empty array.
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.empty(cfg, dtshape)


def arange(
    dtshape: DTShape, start=None, stop=None, step=None, cfg: ia.Config = None, **kwargs
) -> ia.IArray:
    """Return evenly spaced values within a given interval.

    `dtshape`, `cfg` and `kwargs` are the same than for :func:`empty`.

    `start`, `stop`, `step` are the same as in np.arange.

    Returns
    -------
    IArray
        The new array.

    See Also
    --------
    empty : Create an empty array.
    """
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


def linspace(
    dtshape: DTShape, start: float, stop: float, cfg: ia.Config = None, **kwargs
) -> ia.IArray:
    """Return evenly spaced numbers over a specified interval.

    `dtshape`, `cfg` and `kwargs` are the same than for :func:`empty`.

    `start`, `stop` are the same as in np.linspace.

    Returns
    -------
    IArray
        The new array.

    See Also
    --------
    empty : Create an empty array.
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.linspace(cfg, start, stop, dtshape)


def zeros(dtshape: DTShape, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a new array of given shape and type, filled with zeros.

    `dtshape`, `cfg` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    IArray
        The new array.

    See Also
    --------
    empty : Create an empty array.
    ones : Create an array filled with ones.
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.zeros(cfg, dtshape)


def ones(dtshape: DTShape, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a new array of given shape and type, filled with ones.

    `dtshape`, `cfg` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    IArray
        The new array.

    See Also
    --------
    empty : Create an empty array.
    zeros : Create an array filled with zeros.
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.ones(cfg, dtshape)


def full(dtshape: DTShape, fill_value: float, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Return a new array of given shape and type, filled with `fill_value`.

    `dtshape`, `cfg` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    IArray
        The new array.

    See Also
    --------
    empty : Create an empty array.
    zeros : Create an array filled with zeros.
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.full(cfg, fill_value, dtshape)
