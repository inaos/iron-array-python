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
from iarray import py2llvm


# The main expression class
class Expr(ext.Expression):
    """A class that is meant to hold an expression.

    This is not meant to be called directly from user space.

    See Also
    --------
    expr_from_string
    expr_from_udf
    """

    def __init__(self, shape, cfg=None, **kwargs):
        if cfg is None:
            cfg = ia.get_config()

        with ia.config(cfg=cfg, shape=shape, **kwargs) as cfg:
            dtshape = ia.DTShape(shape, cfg.dtype)
            self.cfg = cfg
            super().__init__(self.cfg)
            super().bind_out_properties(dtshape, cfg.store)

    def eval(self) -> ia.IArray:
        """Evaluate the expression in self.

        Returns
        -------
        IArray
            The output array.
        """
        return super().eval()


def check_inputs(inputs: list):
    first_input = inputs[0]
    for input_ in inputs[1:]:
        if first_input.shape != input_.shape:
            raise ValueError("Inputs should have the same shape")
        if first_input.dtype != input_.dtype:
            raise TypeError("Inputs should have the same dtype")
    return first_input.shape, first_input.dtype


def expr_from_string(sexpr: str, inputs: dict, cfg: ia.Config = None, **kwargs) -> Expr:
    """Create an `Expr` instance from a expression in string form.

    Parameters
    ----------
    sexpr : str
        An expression in string format.
    inputs : dict
        Map of variables in `sexpr` to actual arrays.
    cfg : Config
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config
        dataclass that should override the current configuration.

    Returns
    -------
    Expr
        An expression ready to be evaluated via :func:`Expr.eval`.
    """
    shape, dtype = check_inputs(list(inputs.values()))
    kwargs["dtype"] = dtype
    expr = Expr(shape=shape, cfg=cfg, **kwargs)
    for i in inputs:
        expr.bind(i, inputs[i])
    expr.compile(sexpr)
    return expr


def expr_from_udf(udf: py2llvm.Function, inputs: list, cfg=None, dtshape=None, **kwargs) -> Expr:
    """Create an `Expr` instance from an UDF function.

    Parameters
    ----------
    udf : py2llvm.Function
        A User Defined Function.
    inputs : list
        List of arrays whose values are passed as arguments, after the output,
        to the UDF function.
    cfg : Config
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config
        dataclass that should override the current configuration.

    Returns
    -------
    Expr
        An expression ready to be evaluated via :func:`Expr.eval`.
    """
    shape, dtype = check_inputs(inputs)
    kwargs["dtype"] = dtype
    expr = Expr(shape=shape, cfg=cfg, **kwargs)
    for i in inputs:
        expr.bind("", i)
    expr.compile_udf(udf)
    return expr
