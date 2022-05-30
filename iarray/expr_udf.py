import ast
import math

import numpy as np

import iarray as ia
from iarray import udf
from iarray.py2llvm.py2llvm import MATH_FUNCS

try:
    # Python 3.9
    ast.unparse
except AttributeError:
    # Python 3.8
    from ast_decompiler import decompile
else:

    def decompile(tree):
        return ast.unparse(tree)


def name(id, ctx=ast.Load()):
    return ast.Name(id, ctx=ctx)


def constant(value):
    return ast.Constant(value, kind=None)


def For(dim, ndim, body):
    body = [For(dim + 1, ndim, body)] if dim < ndim - 1 else body
    return ast.For(
        target=name(f"i{dim}", ctx=ast.Store()),
        iter=ast.Call(
            func=name("range"),
            args=[
                ast.Subscript(
                    value=ast.Attribute(
                        value=name("out"),
                        attr="shape",
                        ctx=ast.Load(),
                    ),
                    slice=ast.Index(value=constant(dim)),
                    ctx=ast.Load(),
                )
            ],
            keywords=[],
        ),
        body=body,
        orelse=[],
    )


class Transformer(ast.NodeTransformer):
    def __init__(self, args):
        self.args = args
        # The function arguments
        dtype_map = {
            np.float32: "udf.float32",
            np.float64: "udf.float64",
            np.int8: "udf.int8",
            np.int16: "udf.int16",
            np.int32: "udf.int32",
            np.int64: "udf.int64",
            float: "udf.float64",
            int: "udf.float64",  # FIXME Should be int64
        }
        self.func_args = []
        for key, value in args.items():
            if isinstance(value, ia.IArray):
                ndim = value.ndim
                dtype = dtype_map[value.dtype]
                annotation = ast.parse(f"udf.Array({dtype}, {ndim})")
            else:
                dtype = dtype_map[type(value)]
                annotation = ast.parse(dtype)
            self.func_args.append(ast.arg(key, annotation=annotation))

        # The output is the first argument
        # FIXME output name is hardcoded, may conflict with expression names
        # FIXME Cast args types to find out type
        annotation = ast.parse(f"udf.Array({dtype}, {ndim})")
        self.func_args.insert(0, ast.arg("out", annotation))

        # Keep the ndim, and the index used to access the arrays
        self.ndim = ndim
        self.index = ast.Tuple(
            elts=[name(f"i{i}") for i in range(ndim)],
            ctx=ast.Load(),
        )

    def visit_Module(self, node):
        self.generic_visit(node)
        return ast.Module(
            body=[
                ast.FunctionDef(
                    name="f",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=self.func_args,
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[],
                    ),
                    body=[
                        For(0, self.ndim, node.body),
                        ast.Return(value=constant(0)),
                    ],
                    decorator_list=[],
                    returns=None,
                )
            ],
            type_ignores=[],
        )

    def visit_Call(self, node):
        self.generic_visit(node)

        # Translate negative(x) to math.copysign(x, -1.0)
        # https://github.com/inaos/iron-array/issues/559
        if isinstance(node.func, ast.Name):
            if node.func.id in {"negative", "negate"}:
                node.func = ast.Attribute(
                    value=name("math"),
                    attr="copysign",
                    ctx=node.func.ctx,
                )
                node.args.append(constant(-1.0))

        return node

    def visit_Expr(self, node):
        self.generic_visit(node)
        return ast.Assign(
            targets=[
                ast.Subscript(
                    value=name("out"),
                    slice=ast.Index(value=self.index),
                    ctx=ast.Store(),
                )
            ],
            value=node.value,
        )

    def visit_Name(self, node):
        # Translate math function names from those used in mingjugg to those
        # used in Python's math library
        translate_map = {
            "abs": "fabs",
            "absolute": "fabs",
            "arccos": "acos",
            "arcsin": "asin",
            "arctan": "atan",
            "arctan2": "atan2",
            "power": "pow",
        }
        node_id = translate_map.get(node.id, node.id)

        # Math functions
        if node_id in MATH_FUNCS:
            return ast.Attribute(
                value=name("math"),
                attr=node_id,
                ctx=node.ctx,
            )

        # Access to arrays
        arg = self.args.get(node_id)
        if isinstance(arg, ia.IArray):
            return ast.Subscript(
                value=node,
                slice=ast.Index(value=self.index),
                ctx=node.ctx,
            )

        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        slice = node.slice

        # Python 3.8
        if isinstance(slice, ast.Index):
            slice = slice.value

        if isinstance(slice, (ast.UnaryOp, ast.BoolOp, ast.Compare)):
            return ast.IfExp(
                test=slice,
                body=node.value,
                orelse=constant(math.nan),
            )

        return node


def expr_udf(expr, args, cfg=None, debug=0, **kwargs):
    # There must be at least 1 argument
    assert len(args) > 0

    # Split input arrays from input scalars
    arrays = []
    scalars = []
    for value in args.values():
        if isinstance(value, ia.IArray):
            arrays.append(value)
        else:
            scalars.append(value)

    # From the string expression produce the udf function source
    tree = ast.parse(expr)  # AST of the input expression
    tree = Transformer(args).visit(tree)  # AST of the UDF function
    ast.fix_missing_locations(tree)
    source = decompile(tree)  # Source code of the UDF function
    if debug > 0:
        print(source)

    # The Python function
    exec(source, globals(), locals())
    py_func = locals()["f"]

    # The UDF function
    udf_func = udf.jit(py_func, ast=tree, debug=debug)

    # The IArray expression
    ia_expr = ia.expr_from_udf(udf_func, arrays, scalars, cfg=cfg, **kwargs)
    return ia_expr
