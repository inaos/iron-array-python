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
import os
import shutil


zarr_to_iarray_dtypes = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
}


class IllegalArgumentError(ValueError):
    pass


def _check_access_mode(urlpath, mode, update=False):
    """
    Based on `urlpath` and `mode`, remove the possible underlying storage.

    Call this function only when modifying/creating an array.
    """
    supported_modes = [b"r", b"r+", b"w-", b"w", b"a"]
    mode = mode.encode("utf-8") if isinstance(mode, str) else mode
    supported = any(x == mode for x in supported_modes)
    if not supported:
        raise NotImplementedError("The mode is not supported yet.")
    if mode == b"r":
        raise IOError("Cannot do the requested operation with the actual mode.")
    if mode == b"r+" and not update:
        raise IOError("Cannot do the requested operation with the actual mode.")
    if urlpath is not None:
        if mode == b"w":
            if not update:
                ia.remove_urlpath(urlpath)
        elif os.path.exists(urlpath) and mode == b"w-":
            raise IOError(
                f"The writing mode cannot overwrite the already existing array '{urlpath}'."
            )


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
    """Save an array to a binary file in ironArray `.iarr` format.

    If the file already exists it overwrites it.

    The default for this function is `contiguous=True`.

    `cfg` and `kwargs` are the same than for :func:`IArray.copy`.

    Parameters
    ----------
    iarr : :ref:`IArray`
        The array to save.
    urlpath : str
        The url path to save the array.

    See Also
    --------
    load : Load an array from disk.
    open : Open an array from disk.
    """
    ia.remove_urlpath(urlpath)
    if kwargs.get("contiguous", None) is None and (cfg is None or cfg.contiguous is None):
        kwargs = dict(kwargs, contiguous=True)
    iarr.copy(cfg=cfg, urlpath=urlpath, **kwargs)


def load(urlpath: str, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Open an array from a binary file in ironArray `.iarr` format and load data into memory.

    The default for this function is `contiguous=False` and `mode='a'`.

    `cfg` and `kwargs` are the same than for :func:`IArray.copy`.

    Parameters
    ----------
    urlpath : str
        The url path to read.

    Returns
    -------
    :ref:`IArray`
        The new loaded array.

    See Also
    --------
    save : Save an array to disk.
    """
    if kwargs.get("contiguous", None) is None and (cfg is None or cfg.contiguous is None):
        kwargs = dict(kwargs, contiguous=False)
    if kwargs.get("mode", None) is not None:
        iarr = ia.open(urlpath, mode=kwargs.get("mode", None))
    elif cfg is not None:
        iarr = ia.open(urlpath, mode=cfg.mode)
    else:
        iarr = ia.open(urlpath)
    return iarr.copy(cfg=cfg, **kwargs)


def open(urlpath: str, mode="a") -> ia.IArray:
    """Open an array from a binary file in ironArray `.iarr` format.

    The array data will lazily be read when necessary.

    Parameters
    ----------
    urlpath : str
        The url path to read.
    mode : str
        The open mode. This parameter supersedes the mode in the default :class:`Config`.

    Returns
    -------
    :ref:`IArray`
        The new opened array.

    See Also
    --------
    save : Save an array to disk.
    """
    cfg = ia.get_config_defaults()
    if not os.path.exists(urlpath):
        raise IOError("The file does not exist.")
    with ia.config(cfg=cfg, mode=mode) as cfg:
        return ext.open(cfg, urlpath)


# TODO: are cfg and kwargs needed here?
def iarray2numpy(iarr: ia.IArray, cfg: ia.Config = None, **kwargs) -> np.ndarray:
    """Convert an ironArray array into a NumPy array.

    `cfg` and `kwargs` are the same than for :func:`empty`.

    Parameters
    ----------
    iarr : :ref:`IArray`
        The array to convert.

    Returns
    -------
    out: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        The new NumPy array.

    See Also
    --------
    numpy2iarray : Convert a NumPy array into an ironArray array.
    """
    if cfg is None:
        cfg = ia.get_config_defaults()

    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.iarray2numpy(cfg, iarr)


def numpy2iarray(arr: np.ndarray, cfg: ia.Config = None, **kwargs) -> ia.IArray:
    """Convert a NumPy array into an ironArray array.

    `cfg` and `kwargs` are the same than for :func:`empty`.

    Parameters
    ----------
    arr : `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        The array to convert.

    Returns
    -------
    :ref:`IArray`
        The new ironArray array.

    See Also
    --------
    iarray2numpy : Convert an ironArray array into a NumPy array.
    """

    if np.dtype(arr.dtype) in [np.dtype(d) for d in zarr_to_iarray_dtypes]:
        dtype = arr.dtype
    else:
        raise NotImplementedError("Only float32 and float64 types are supported for now")

    kwargs["dtype"] = dtype

    if cfg is None:
        cfg = ia.get_config_defaults()

    if not arr.flags["C_CONTIGUOUS"]:
        # For the conversion we need a *C* contiguous array
        arr = arr.copy(order="C")

    with ia.config(shape=arr.shape, cfg=cfg, **kwargs) as cfg:
        dtshape = ia.DTShape(arr.shape, cfg.dtype)
        return ext.numpy2iarray(cfg, arr, dtshape)


# File system utilities
def remove_urlpath(urlpath):
    """Permanently remove the file or the directory given by `urlpath`.

    Parameters
    ----------
    urlpath: String
        The path of the directory or file.

    Returns
    -------
    None
    """
    if urlpath is not None:
        if os.path.exists(urlpath):
            if os.path.isdir(urlpath):
                shutil.rmtree(urlpath)
            else:
                os.remove(urlpath)
