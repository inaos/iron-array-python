###########################################################################################
# Copyright ironArray SL 2021.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of ironArray SL
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement.
###########################################################################################
from collections.abc import MutableMapping
from typing import Optional
import re

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
            cfg = ia.get_config_defaults()

        default_shapes = check_expr_config(cfg, **kwargs)
        with ia.config(cfg=cfg, shape=shape, **kwargs) as cfg:
            dtshape = ia.DTShape(shape, cfg.dtype)
            self.cfg = cfg
            super().__init__(self.cfg)
            super().bind_out_properties(dtshape)
            if default_shapes:
                # Set cfg chunks and blocks to None to detect that we want the default shapes when evaluating
                self.cfg.chunks = None
                self.cfg.blocks = None

    def eval(self) -> ia.IArray:
        """Evaluate the expression in self.

        Returns
        -------
        :ref:`IArray`
            The output array.
        """
        return super().eval()


def check_inputs(inputs: list, shape):
    if inputs:
        first_input = inputs[0]
        for input_ in inputs[1:]:
            if first_input.shape != input_.shape:
                raise ValueError("Inputs should have the same shape")
            if first_input.dtype != input_.dtype:
                raise TypeError("Inputs should have the same dtype")
        return first_input.shape, first_input.dtype
    else:
        cfg = ia.get_config_defaults()
        if shape is None:
            raise AttributeError("A shape is needed")
        return shape, cfg.dtype


# Check that inputs values are compatible with operands in expression
def check_expr(sexpr: str, inputs: dict):
    vars_in_expr = expr_get_operands(sexpr)
    if not set(vars_in_expr).issubset(set(inputs.keys())):
        raise ValueError(f"Some operands in expression {vars_in_expr} are not in input keys")
    for var in vars_in_expr:
        if not isinstance(inputs[var], ia.IArray):
            raise ValueError(f"Operand {var} is not an IArray instance")


def expr_from_string(sexpr: str, inputs: dict, cfg: ia.Config = None, **kwargs) -> Expr:
    """Create an :class:`Expr` instance from an expression in string form.

    Parameters
    ----------
    sexpr : str
        An expression in string form.
    inputs : dict
        Map of variables in `sexpr` to actual arrays.
    cfg : :class:`Config`
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :class:`Expr`
        An expression ready to be evaluated via :func:`Expr.eval`.

    See Also
    --------
    expr_from_udf
    """
    check_expr(sexpr, inputs)
    with ia.config(cfg, **kwargs):
        shape, dtype = check_inputs(list(inputs.values()), None)
    kwargs["dtype"] = dtype
    expr = Expr(shape=shape, cfg=cfg, **kwargs)
    for i in inputs:
        expr.bind(i, inputs[i])
    expr.compile(sexpr)
    return expr


def expr_from_udf(
    udf: py2llvm.Function,
    inputs: list,
    params: Optional[list] = None,
    shape=None,
    cfg=None,
    **kwargs
) -> Expr:
    """Create an :class:`Expr` instance from an UDF function.

    Parameters
    ----------
    udf : py2llvm.Function
        A User Defined Function.
    inputs : list
        List of arrays whose values are passed as arguments, after the output,
        to the UDF function.
    params : list
        List user parameters, other than the input arrays, passed to the user
        defined function.
    cfg : :class:`Config`
        The configuration for running the expression.
        If None (default), global defaults are used.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :class:`Expr`
        An expression ready to be evaluated via :func:`Expr.eval`.

    See Also
    --------
    expr_from_string
    """
    if params is None:
        params = []

    # Build expression
    with ia.config(cfg, shape=shape, **kwargs):
        shape, dtype = check_inputs(inputs, shape)

    kwargs["dtype"] = dtype
    expr = Expr(shape=shape, cfg=cfg, **kwargs)

    # Bind input arrays
    for i in inputs:
        expr.bind("", i)

    # Bind input scalars
    sig_params = udf.py_signature.parameters[1:] # The first param is the output array
    sig_params = sig_params[len(inputs):] # Next come the input arrays
    assert len(params) == len(sig_params) # What is left are the user params (scalars)

    for value, sig_param in zip(params, sig_params):
        expr.bind_param(value, sig_param.type)

    # Compile
    expr.compile_udf(udf)
    return expr


def check_expr_config(cfg=None, **kwargs):
    # Check if the chunks and blocks are explicitly set
    default_shapes = False
    if (cfg is not None and cfg.chunks is None and cfg.blocks is None) or cfg is None:
        shape_params = {"chunks", "blocks"}
        if kwargs != {}:
            not_kw_shapes = all(x not in kwargs for x in shape_params)
            if not_kw_shapes:
                default_shapes = True
        else:
            default_shapes = True
    return default_shapes


# Compile the regular expression to find operands
# The expression below does not detect functions like 'lib.func()'
# operands_regex = r"\w+(?=\()|((?!0)|[-+]|(?=0+\.))(\d*\.)?\d+(e\d+)?|(\w+)"
# See https://regex101.com/r/ASRG5J/1
operands_regex = r"((\w\.*)+)(\()|((?!0)|[-+]|(0+\.))(\d*\.)?\d+(e\d+)?|(\w+)"
operands_regex_compiled = re.compile(operands_regex)
def expr_get_operands(sexpr):
    """Return a tuple with the operands of an expression in string form.

    Parameters
    ----------
    sexpr : str
        An expression in string form.

    Returns
    -------
    tuple
        The operands list.
    """
    m2 = operands_regex_compiled.findall(sexpr)
    # We are interested in the last group (operands) only
    # print("->", list(m2))
    operands = set(g[-1] for g in m2 if g[-1] != '')
    return tuple(sorted(operands))


class UdfRegistry(MutableMapping):

    def __init__(self):
        self.libs = {}
        self.libs_funcs = {}

    def __getitem__(self, name: str):
        """
        Get the :paramref:`name` library.

        Parameters
        ----------
        name : str or byte string
            The name of the library to return.

        Returns
        -------
        object : set
            A set with all the UDF functions in :paramref:`name` lib.
        """
        if name in self.libs:
            return self.libs_funcs[name]
        raise ValueError(f"'{name}' is not a registered library")

    def __setitem__(self, name: str, udf_func: py2llvm.Function):
        """Add a UDF function to :paramref:`name` library.

        Parameters
        ----------
        name : str or byte string
            The name of the library.
        udf_func : py2llvm.Function
            The UDF function.

        Returns
        -------
        None
        """
        if name not in self.libs:
            self.libs[name] = ext.UdfLibrary(name)
            self.libs_funcs[name] = set()
        if udf_func.name in self.libs_funcs[name]:
            raise ValueError(f"UDF func '{udf_func.name}' already registered in library '{name}'")
        self.libs[name].register_func(udf_func)
        self.libs_funcs[name].add(udf_func)

    def __delitem__(self, name: str):
        """Delete the attr given by :paramref:`name`.

        Parameters
        ----------
        name : str or byte string
            The name of the attr.
        """
        self.libs[name].dealloc()
        del self.libs[name]
        del self.libs_funcs[name]

    def __iter__(self):
        """
        Iterate over all the names of the registered libs.
        """
        for name in self.libs:
            yield name

    def iter_funcs(self, name: str):
        """
        Iterate over all UDF funcs registered in :paramref:`name` lib.

        Parameters
        ----------
        name : str
            The name of the library

        Returns
        -------
        Iterator over all funcs under :paramref:`name` lib.
        """
        for func in self.libs_funcs[name]:
            yield func

    def iter_all_func_names(self):
        """
        Iterate over all UDF func names registered in all libs.

        Returns
        -------
        Iterator over all registered UDF funcs in the `lib_name.func_name` form.
        """
        for name in self.libs:
            for func in self.libs_funcs[name]:
                yield ".".join([name, func.name])

    def __len__(self):
        return len(self.libs)

    def clear(self):
        """
        Clear all the registered libs and UDF funcs.
        """
        for name in list(self.libs):
            self.__delitem__(name)
        self.libs = {}
        self.libs_funcs = {}
