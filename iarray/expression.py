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
import re
from typing import Optional

from llvmlite import ir

import iarray as ia
from iarray import iarray_ext as ext
from iarray import py2llvm
from iarray import udf
from iarray.expr_udf import expr_udf


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
        self.input_refs = {}  # keep references to some inputs alive!

    def bind(self, var, value, keep_ref=False):
        """
        Bind var names to input arrays.

        Params
        ------
        var : str
            The name of the variable in the expression.
        value : :ref:`IArray`
            The actual array that is attached to the variable.
        """
        if keep_ref:
            self.input_refs[var] = value  # add a reference to this input
        return super().bind(var, value)

    def eval(self) -> ia.IArray:
        """Evaluate the expression in self.

        Returns
        -------
        :ref:`IArray`
            The output array.
        """
        iarr = super().eval()
        # We don't want to free references to new arrays coming from scalars
        # This would prevent to reuse the expression instance in e.g. bench loops.
        # self.input_refs = {}   # free internal reference to inputs
        a = iarr
        a.np_dtype = self.cfg.np_dtype
        return a


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
    operands = tuple(sorted(set(g[-1] for g in m2 if g[-1] != "")))
    regular_funcs = tuple(sorted(set(g[0] for g in m2 if g[0] != "" and "." not in g[0])))
    udf_funcs = tuple(sorted(set(g[0] for g in m2 if g[0] != "" and "." in g[0])))
    return operands, regular_funcs, udf_funcs


# Check validity for operands, regular functions and udf functions in expression
def check_expr(sexpr: str, inputs: dict):
    ops_in_expr, regular_funcs_in_expr, udf_funcs_in_expr = expr_get_ops_funcs(sexpr)

    # Operands
    if not set(ops_in_expr).issubset(set(inputs.keys())):
        ops_not_in_expr = tuple(set(ops_in_expr) - set(inputs.keys()))
        raise ValueError(f"Some operands in expression {ops_not_in_expr} are not in input keys")
    for op in ops_in_expr:
        if not isinstance(inputs[op], ia.IArray):
            raise ValueError(f"Operand {op} is not an IArray instance")

    # Regular functions
    if not set(regular_funcs_in_expr).issubset(set(ia.MATH_FUNC_LIST)):
        raise ValueError(
            f"Some regular funcs in expression {regular_funcs_in_expr} are not allowed"
        )

    # UDF functions
    reg_funcs = set(ia.udf_registry.iter_all_func_names())
    if not set(udf_funcs_in_expr).issubset(reg_funcs):
        raise ValueError(
            f"Some UDF funcs in expression {udf_funcs_in_expr} are not registered yet"
        )

    return ops_in_expr


def check_inputs_string(inputs: dict, cfg: ia.Config, minjugg: bool = False):
    """
    Check the inputs for a expression in string form.

    If scalars are found, they are broadcasted to the final shape.

    Parameters
    ----------
    inputs: dict
        A map for operand names and arrays or scalars.

    cfg: ia.Config
        The default config for new arrays from scalars.

    Returns
    -------
    tuple
        A tuple of shape, dtype and the updated dict of inputs
    """
    arrays = dict()
    scalars = dict()
    for iname, ivalue in inputs.items():
        if hasattr(ivalue, "shape"):
            if ivalue.shape != ():
                arrays[iname] = ivalue
            else:
                # Promote a 0-dim array to 1-dim
                a = ia.full(shape=(1,), fill_value=ivalue.data[()], cfg=ivalue.cfg, urlpath=None,
                            chunks=(1,), blocks=(1,))
                inputs[iname] = a
                arrays[iname] = a
        else:
            scalars[iname] = ivalue
    if len(arrays) == 0:
        raise ValueError(
            "You need to pass at least one array.  Use ia.empty() if values are not really needed."
        )

    # Get the shape and dtype for array operands
    larrays = list(arrays.values())
    first_array = larrays[0]
    for array in larrays[1:]:
        if first_array.shape != array.shape:
            raise ValueError("Arrays in inputs should have the same shape")
        if first_array.dtype != array.dtype:
            raise TypeError("Arrays in inputs should have the same dtype")
    shape, chunks, blocks, dtype = (
        first_array.shape,
        first_array.chunks,
        first_array.blocks,
        first_array.dtype,
    )

    # Now convert the scalars to arrays with the proper shape and dtype
    new_inputs = {}
    for skey, svalue in scalars.items():
        if minjugg:
            # minjugg has not been tested for handling scalars as operands
            # Using the same chunks and blocks maximizes the chances to use ITERBLOSC
            new_inputs[skey] = ia.full(
                shape=shape, fill_value=svalue, dtype=dtype, cfg=cfg, chunks=chunks, blocks=blocks
            )
        else:
            # The py2llvm backend does support handling scalars as operands
            new_inputs[skey] = svalue

    return shape, dtype, arrays, new_inputs


def expr_from_string(
    sexpr: str, inputs: dict, cfg: ia.Config = None, debug: int = 0, **kwargs
) -> Expr:
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
    with ia.config(cfg, **kwargs) as cfg:
        shape, dtype, array_inputs, new_inputs = check_inputs_string(inputs, cfg, minjugg=False)
        np_dtype = cfg.np_dtype
    kwargs["dtype"] = dtype
    kwargs["np_dtype"] = np_dtype
    # The lines below use the evaluator from minjugg
    # check_expr(sexpr, operands)
    # expr = Expr(shape=shape, cfg=cfg, **kwargs)
    # for k, v in array_inputs.items():
    #     expr.bind(k, v)
    # for k, v in new_inputs.items():
    #     # These are arrays created anew.  Keep the reference to them.
    #     expr.bind(k, v, keep_ref=True)
    # expr.compile(sexpr)
    # The next uses the expr -> UDF machinery, which has support for masks and others bells and whistles
    operands = {**array_inputs, **new_inputs}
    with ia.config(cfg, **kwargs) as cfg:
        expr = expr_udf(sexpr, operands, cfg=cfg, debug=debug)
    return expr


def check_inputs_udf(inputs: list):
    if len(inputs) == 0:
        raise ValueError(
            "You need to pass at least one array.  Use ia.empty() if values are not really needed."
        )
    first_input = inputs[0]
    for input_ in inputs[1:]:
        if first_input.shape != input_.shape:
            raise ValueError("Inputs should have the same shape")
        if first_input.dtype != input_.dtype:
            raise TypeError("Inputs should have the same dtype")
    return first_input.shape, first_input.dtype


def expr_from_udf(
    udf: py2llvm.Function,
    inputs: list,
    params: Optional[list] = None,
    shape=None,
    cfg=None,
    **kwargs,
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
    shape : Sequence
        The shape for the output array.  If None, the value is derived from the inputs.
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
    with ia.config(cfg, **kwargs) as cfg:
        np_dtype = cfg.np_dtype
        if inputs:
            shape, dtype = check_inputs_udf(inputs)
        else:
            dtype = cfg.dtype
    kwargs["dtype"] = dtype
    kwargs["np_dtype"] = np_dtype
    expr = Expr(shape=shape, cfg=cfg, **kwargs)

    # Bind input arrays
    for i in inputs:
        expr.bind("", i)

    # Bind input scalars
    sig_params = udf.py_signature.parameters[1:]  # The first param is the output array
    sig_params = sig_params[len(inputs) :]  # Next come the input arrays
    assert len(params) == len(sig_params)  # What is left are the user params (scalars)

    for value, sig_param in zip(params, sig_params):
        expr.bind_param(value, sig_param.type)

    # Compile
    expr.compile_udf(udf)
    return expr


def check_expr_config(cfg=None, **kwargs):
    # Check that the chunks and blocks are explicitly set
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
        The list of operands.
    """
    return expr_get_ops_funcs(sexpr)[0]


# Accessor for the scalar UDF functions (for lazy expressions)
class UFunc:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return ia.LazyExpr(new_op=(self, self.name, [*args]))


# Accessor for the default scalar UDF library (for lazy expressions)
class ULib:
    def __init__(self, dfltlib):
        self.dfltlib = dfltlib

    def __getattr__(self, name):
        full_name = f"{self.dfltlib}.{name}"
        try:
            func = ia.udf_lookup_func(full_name)
        except:
            raise AttributeError(f"{full_name} scalar UDF function not found")
        return UFunc(full_name)


class UdfLibrary(ext.UdfLibrary):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.functions = {}

    def register_func(self, function):
        name = function.name
        if name in self.functions:
            raise ValueError(f"UDF func '{name}' already registered in library '{self.name}'")

        super().register_func(function)
        self.functions[name] = function

    def __getattr__(self, name):
        full_name = f"{self.name}.{name}"
        try:
            address = ext.udf_lookup_func(full_name)
        except ia.IArrayError:
            raise ValueError(f"'{full_name}' is not a registered UDF function")

        # TODO I think it would be simpler and better to instead use
        # llvmlite.binding.add_symbol(name, address), but I discovered this a
        # bit too late.

        # Convert function address (int) to IR function pointer
        address = ir.Constant(udf.int64, address)
        f_type = self.functions[name].ir_function_type
        f_type_ptr = f_type.as_pointer()
        f_ptr = address.inttoptr(f_type_ptr)
        f_ptr.function_type = f_type

        return f_ptr


class UdfRegistry(MutableMapping):
    def __init__(self):
        self.libs = {}  # name: <UdfLibrary>
        self.func_addr = {}  # name.fname: address (int)

    def __getitem__(self, name: str):
        """
        Get the :paramref:`name` library.

        Parameters
        ----------
        name : str or byte string
            The name of the library to return.

        Returns
        -------
        object : dict
            A dict with all the UDF functions in :paramref:`name` lib.
            The key is the function name and the value is an instance of
            udf.Function
        """
        if name in self.libs:
            return self.libs[name].functions
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
            self.libs[name] = UdfLibrary(name)
        self.libs[name].register_func(udf_func)
        full_func_name = f"{name}.{udf_func.name}"
        self.func_addr[full_func_name] = ia.udf_lookup_func(full_func_name)

    def __delitem__(self, name: str):
        """Delete the attr given by :paramref:`name`.

        Parameters
        ----------
        name : str or byte string
            The name of the attr.
        """
        for func_name in self.libs[name].functions:
            del self.func_addr[f"{name}.{func_name}"]
        self.libs[name].dealloc()
        del self.libs[name]

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
        yield from self.libs[name].functions.values()

    def iter_all_func_names(self):
        """
        Iterate over all UDF func names registered in all libs.

        Returns
        -------
        Iterator over all registered UDF funcs in the `lib_name.func_name` form.
        """
        for name in self.libs:
            for func_name in self.libs[name].functions:
                yield f"{name}.{func_name}"

    def __len__(self):
        return len(self.libs)

    def clear(self):
        """
        Clear all the registered libs and UDF funcs.
        """
        for name in list(self.libs):
            self.__delitem__(name)
        self.libs = {}
        self.func_addr = {}
