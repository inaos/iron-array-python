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
from iarray import py2llvm


def cmp_arrays(a, b, success=None):
    """Quick and dirty comparison between arrays `a` and `b`.

    The arrays `a` and `b` are converted internally to numpy arrays, so
    this can require a lot of memory.  This is mainly used for quick testing.

    If success, and the string passed in `success` is not None, it is printed.
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


#
# Expressions
#


def check_inputs(inputs: list):
    first_input = inputs[0]
    for input_ in inputs[1:]:
        if first_input.shape != input_.shape:
            raise ValueError("Inputs should have the same shape")
        if first_input.dtype != input_.dtype:
            raise TypeError("Inputs should have the same dtype")
    return first_input.dtshape


def expr_from_string(sexpr: str, inputs: dict, cfg: ia.Config = None, **kwargs):
    """Create an `Expr` instance from a expression in string form.

    Parameters
    ----------
    sexpr : str
        An expression in string format.
    inputs : dict
        Map of variables in `sexpr` to actual arrays.
    cfg : ia.Config
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the ia.Config
        dataclass that should override the current configuration.

    Returns
    -------
    ia.Expr
        An expression ready to be evaluated via `.eval()`.
    """
    dtshape = check_inputs(list(inputs.values()))
    expr = Expr(dtshape=dtshape, cfg=cfg, **kwargs)
    for i in inputs:
        expr.bind(i, inputs[i])
    expr.compile(sexpr)
    return expr


def expr_from_udf(udf: py2llvm.Function, inputs: list, cfg=None, **kwargs):
    """Create an `Expr` instance from an UDF function.

    Parameters
    ----------
    udf : py2llvm.Function
        A User Defined Function.
    inputs : list
        List of arrays whose values are passed as arguments, after the output,
        to the UDF function.
    cfg : ia.Config
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the ia.Config
        dataclass that should override the current configuration.

    Returns
    -------
    ia.Expr
        An expression ready to be evaluated via `.eval()`.
    """
    dtshape = check_inputs(inputs)
    expr = Expr(dtshape=dtshape, cfg=cfg, **kwargs)
    for i in inputs:
        expr.bind("", i)
    expr.compile_udf(udf)
    return expr


# The main expression class
class Expr(ext.Expression):
    """An class that is meant to hold an expression.

    This is not meant to be called directly from user space.

    See Also
    --------
    ia.expr_from_string
    ia.expr_from_udf
    """

    def __init__(self, dtshape, cfg=None, **kwargs):
        with ia.config(cfg=cfg, dtshape=dtshape, **kwargs) as cfg:
            self.cfg = cfg
            super().__init__(self.cfg)
            super().bind_out_properties(dtshape, cfg.storage)

    def eval(self):
        """Evaluate the expression in self.

        Returns
        -------
        ia.IArray
            The output array.
        """
        return super().eval()


# TODO: are cfg and kwargs needed here?
def save(iarr: ia.IArray, filename: str, cfg: ia.Config = None, **kwargs):
    """Save an array to a binary file in ironArray ``.iarray`` format.

    `cfg` and `kwargs` are the same than for ia.empty.

    Parameters
    ----------
    iarr : ia.IArray
        The array to save.
    filename : str
        The file name to save the array.

    See Also
    --------
    ia.load : Load an array from disk.
    """
    with ia.config(cfg=cfg, **kwargs) as cfg:
        ext.save(cfg, iarr, filename)


def load(filename: str, load_in_mem: bool = False, cfg: ia.Config = None, **kwargs):
    """Load an array from a binary file in ironArray ``.iarray`` format.

    `cfg` and `kwargs` are the same than for ia.empty.

    Parameters
    ----------
    filename : str
        The file name to read.
    load_in_mem : bool
        If True, the array is completely loaded in-memory.  If False (default),
        the array will lazily be read when necessary.

    Returns
    -------
    IArray
        The new loaded array.

    See Also
    --------
    ia.save : Save an array to disk.
    """
    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.load(cfg, filename, load_in_mem)


# TODO: are cfg and kwargs needed here?
def iarray2numpy(iarr: ia.IArray, cfg: ia.Config = None, **kwargs):
    """Convert an ironArray array into a NumPy array.

    `cfg` and `kwargs` are the same than for ia.empty.

    Parameters
    ----------
    iarr : ia.IArray
        The array to convert.

    Returns
    -------
    np.ndarray
        The new NumPy array.

    See Also
    --------
    ia.numpy2iarray : Convert a NumPy array into an ironArray array.
    """
    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.iarray2numpy(cfg, iarr)


def numpy2iarray(arr: np.ndarray, cfg: ia.Config = None, **kwargs):
    """Convert a NumPy array into an ironArray array.

    `cfg` and `kwargs` are the same than for ia.empty.

    Parameters
    ----------
    arr : np.ndarray
        The array to convert.

    Returns
    -------
    ia.IArray
        The new ironArray array.

    See Also
    --------
    ia.iarray2numpy : Convert an ironArray array into a NumPy array.
    """
    if arr.dtype == np.float64:
        dtype = np.float64
    elif arr.dtype == np.float32:
        dtype = np.float32
    else:
        raise NotImplementedError("Only float32 and float64 types are supported for now")

    dtshape = ia.DTShape(arr.shape, dtype)
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.numpy2iarray(cfg, arr, dtshape)
