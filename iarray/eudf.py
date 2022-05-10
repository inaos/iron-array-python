import ast
from ast_decompiler import decompile

import iarray as ia
from iarray import udf


def name(id, ctx=ast.Load()):
    return ast.Name(id, ctx=ctx)


def For(dim, ndim, body):
    body = [For(dim + 1, ndim, body)] if dim < ndim - 1 else body
    return ast.For(
        target=name(f'i{dim}'),
        iter=ast.Call(
            func=name('range'),
            args=[
                ast.Subscript(
                    value=ast.Attribute(
                        value=name('out'),
                        attr='shape',
                        ctx=ast.Load(),
                    ),
                    slice=ast.Index(value=ast.Constant(dim)),
                    ctx=ast.Load()
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
        # FIXME the datatype is hardcoded
        dtype = 'udf.float64'

        # The function arguments
        self.func_args = []
        for name, value in args.items():
            if isinstance(value, ia.IArray):
                ndim = value.ndim
                annotation = ast.parse(f'udf.Array({dtype}, {ndim})')
            else:
                annotation = ast.parse(dtype)
            self.func_args.append(ast.arg(name, annotation=annotation))

        # The output is the first argument
        # FIXME output name is hardcoded, may conflict with expression names
        annotation = ast.parse(f'udf.Array(udf.float64, {ndim})')
        self.func_args.insert(0, ast.arg('out', annotation))

        # Keep the ndim, and the index used to access the arrays
        self.ndim = ndim
        self.index = ','.join(f'i{i}' for i in range(ndim))


    def visit_Module(self, node):
        self.generic_visit(node)
        return ast.Module(body=[
            ast.FunctionDef(
                name='f',
                args=ast.arguments(
                    posonlyargs=[],
                    args=self.func_args,
                    #arg? vararg,
                    kwonlyargs=[],
                    kw_defaults=[],
                    #arg? kwarg,
                    defaults=[]
                ),
                body=[
                    For(0, self.ndim, node.body),
                    ast.Return(value=ast.Constant(0)),
                ],
                decorator_list=[],
            )
        ])

    def visit_Expr(self, node):
        self.generic_visit(node)
        return ast.Assign(
            targets = [
                ast.Subscript(
                    value=name('out'),
                    slice=ast.Index(value=name(self.index)),
                    ctx=ast.Store(),
                )
            ],
            value = node.value,
        )

    def visit_Name(self, node):
        arg = self.args[node.id]
        if isinstance(arg, ia.IArray):
            return ast.Subscript(
                value=node,
                slice=ast.Index(value=name(self.index)),
                ctx=node.ctx
            )

        # Scalar
        return node


def eudf(expr, args, debug=False):
    # There must be at least 1 argument
    assert len(args) > 0

    # Split input arrays from input scalars
    # TODO Verify all arrays have the same shape
    arrays = []
    scalars = []
    for value in args.values():
        if isinstance(value, ia.IArray):
            arrays.append(value)
        else:
            scalars.append(value)

    # From the string expression produce the udf function source
    tree = ast.parse(expr)               # AST of the input expression
    tree = Transformer(args).visit(tree) # AST of the UDF function
    source = decompile(tree)             # Source code of the UDF function
    if debug:
        print(source)

    # The Python function
    exec(source, globals(), locals())
    py_func = locals()['f']

    # The UDF function
    udf_func = udf.jit(py_func, source=source)

    # The IArray expression
    ia_expr = ia.expr_from_udf(udf_func, arrays, scalars)
    return ia_expr
