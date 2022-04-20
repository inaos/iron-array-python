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


# Compile the regular expression to find operands
# The expression below does not detect functions like 'lib.func()'
# operands_regex = r"\w+(?=\()|((?!0)|[-+]|(?=0+\.))(\d*\.)?\d+(e\d+)?|(\w+)"
# See https://regex101.com/r/ASRG5J/1
operands_regex = r"((\w\.*)+)(\()|((?!0)|[-+]|(0+\.))(\d*\.)?\d+(e\d+)?|(\w+)"
operands_regex_compiled = re.compile(operands_regex)
def expr_get_ops_funcs(sexpr):
    """Return the operands and functions of an expression in string form.

    Parameters
    ----------
    sexpr : str
        An expression in string form.

    Returns
    -------
    tuple
        A tuple of tuples: ((operands), (regular_funcs), (udf_funcs)).
    """
    m2 = operands_regex_compiled.findall(sexpr)
    operands = tuple(sorted(set(g[-1] for g in m2 if g[-1] != '')))
    regular_funcs = tuple(sorted(set(g[0] for g in m2 if g[0] != '' and '.' not in g[0])))
    udf_funcs = tuple(sorted(set(g[0] for g in m2 if g[0] != '' and '.' in g[0])))
    return operands, regular_funcs, udf_funcs


# Check validity for operands, regular functions and udf functions in expression
def check_expr(sexpr: str, inputs: dict):
    ops_in_expr, regular_funcs_in_expr, udf_funcs_in_expr = expr_get_ops_funcs(sexpr)

    # Operands
    if not set(ops_in_expr).issubset(set(inputs.keys())):
        raise ValueError(f"Some operands in expression {ops_in_expr} are not in input keys")
    for op in ops_in_expr:
        if not isinstance(inputs[op], ia.IArray):
            raise ValueError(f"Operand {op} is not an IArray instance")

    # Regular functions
    if not set(regular_funcs_in_expr).issubset(set(ia.MATH_FUNC_LIST)):
        raise ValueError(f"Some regular funcs in expression {regular_funcs_in_expr} are not allowed")

    # UDF functions
    reg_funcs = set(ia.udf_registry.iter_all_func_names())
    if not set(udf_funcs_in_expr).issubset(reg_funcs):
        raise ValueError(f"Some UDF funcs in expression {udf_funcs_in_expr} are not registered yet")


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
    return expr_get_ops_funcs(sexpr)[0]


class UdfRegistry(MutableMapping):

    def __init__(self):
        self.libs = {}
        self.libs_funcs = {}
        self.func_addr = {}

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
        func_name = udf_func.name
        if func_name in self.libs_funcs[name]:
            raise ValueError(f"UDF func '{func_name}' already registered in library '{name}'")
        self.libs[name].register_func(udf_func)
        self.libs_funcs[name].add(udf_func)
        full_func_name = ".".join([name, func_name])
        self.func_addr[full_func_name] = ia.udf_lookup_func(full_func_name)

    def __delitem__(self, name: str):
        """Delete the attr given by :paramref:`name`.

        Parameters
        ----------
        name : str or byte string
            The name of the attr.
        """
        for func in self.libs_funcs[name]:
            full_func_name = ".".join([name, func.name])
            del self.func_addr[full_func_name]
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

    def get_func_addr(self, func_name):
        """
        Return the address of the compiled UDF function.

        Parameters
        ----------
        func_name : str
            The name of the function in `lib_name.func_name` form.

        Returns
        -------
        An integer with the address of the compiled UDF function.  Yes, this is mainly for devs.
        """

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
        self.func_addr = {}
