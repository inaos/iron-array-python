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
import os
import shutil


# TODO: complete support for all possible values of mode
def _check_path_mode(urlpath, mode):
    """
    Based on `urlpath` and `mode`, remove the possible underlying storage.
    """
    if urlpath is not None and mode == b"w" and os.path.exists(urlpath):
        if os.path.isdir(urlpath):
            shutil.rmtree(urlpath)
        else:
            os.remove(urlpath)


def cmp_arrays(a, b, success=None) -> None:
    """Quick and dirty comparison between arrays `a` and `b`.

    The arrays `a` and `b` are converted internally to numpy arrays, so
    this can require a lot of memory.  This is mainly used for quick testing.

    If success, and the string passed in `success` is not None, it is printed.
    If failed, an exception will be raised.
    """
    if type(a) is ia.IArray:
        a = ia.iarray2numpy(a)

    if type(b) is ia.IArray:
        b = ia.iarray2numpy(b)

    if a.dtype == np.float64 and b.dtype == np.float64:
        tol = 1e-14
    else:
        tol = 1e-6
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)

    if success is not None:
        print(success)


# TODO: are cfg and kwargs needed here?
def save(urlpath: str, iarr: ia.IArray, cfg: ia.Config = None, **kwargs) -> None:
    """Save an array to a binary file in ironArray ``.iarray`` format.

    `cfg` and `kwargs` are the same than for :func:`empty`.

    Parameters
    ----------
    iarr : IArray
        The array to save.
    urlpath : str
        The url path to save the array.

    See Also
    --------
    load : Load an array from disk.
    open : Open an array from disk.
    """
    iarr.copy(cfg=cfg, urlpath=urlpath, **kwargs)


def load(filename: str, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Open an array from a binary file in ironArray ``.iarray`` format and load data into memory.

    `cfg` and `kwargs` are the same than for :func:`empty`.

    Parameters
    ----------
    filename : str
        The file name to read.

    Returns
    -------
    IArray
        The new loaded array.

    See Also
    --------
    save : Save an array to disk.
    """

    if cfg is None:
        cfg = ia.get_config()

    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.load(cfg, filename)


def open(filename: str, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Open an array from a binary file in ironArray ``.iarray`` format. The array data will lazily
    be read when necessary.

    `cfg` and `kwargs` are the same than for :func:`empty`.

    Parameters
    ----------
    filename : str
        The file name to read.

    Returns
    -------
    IArray
        The new opened array.

    See Also
    --------
    save : Save an array to disk.
    """
    if cfg is None:
        cfg = ia.get_config()

    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.open(cfg, filename)


# TODO: are cfg and kwargs needed here?
def iarray2numpy(iarr: ia.IArray, cfg: ia.Config = None, **kwargs) -> np.ndarray:
    """Convert an ironArray array into a NumPy array.

    `cfg` and `kwargs` are the same than for :func:`empty`.

    Parameters
    ----------
    iarr : IArray
        The array to convert.

    Returns
    -------
    np.ndarray
        The new NumPy array.

    See Also
    --------
    numpy2iarray : Convert a NumPy array into an ironArray array.
    """
    if cfg is None:
        cfg = ia.get_config()

    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.iarray2numpy(cfg, iarr)


def numpy2iarray(arr: np.ndarray, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Convert a NumPy array into an ironArray array.

    `cfg` and `kwargs` are the same than for :func:`empty`.

    Parameters
    ----------
    arr : np.ndarray
        The array to convert.

    Returns
    -------
    IArray
        The new ironArray array.

    See Also
    --------
    iarray2numpy : Convert an ironArray array into a NumPy array.
    """
    if arr.dtype == np.float64:
        dtype = np.float64
    elif arr.dtype == np.float32:
        dtype = np.float32
    else:
        raise NotImplementedError("Only float32 and float64 types are supported for now")

    kwargs["dtype"] = dtype

    if cfg is None:
        cfg = ia.get_config()

    if arr.base is not None:
        # arr is a view, convert it into a real ndarray
        arr = arr.copy()

    with ia.config(shape=arr.shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(arr.shape, cfg.dtype)
        return ext.numpy2iarray(cfg, arr, dtshape)
