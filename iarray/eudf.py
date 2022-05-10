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

    def __init__(self, ndim):
        self.ndim = ndim
        self.annotation = ast.parse(f'udf.Array(udf.float64, {ndim})')
        self.index = ','.join(f'i{i}' for i in range(ndim))
        # FIXME output name is hardcoded, may conflict with expression names
        self.names = ['out']

    def visit_Module(self, node):
        self.generic_visit(node)
        args = [
            ast.arg(name, annotation=self.annotation)
            for name in self.names
        ]
        return ast.Module(body=[
            ast.FunctionDef(
                name='f',
                args=ast.arguments(
                    posonlyargs=[],
                    args=args,
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
        if node.id not in self.names:
            self.names.append(node.id)

        return ast.Subscript(
            value=name(node.id),
            slice=ast.Index(value=name(self.index)),
            ctx=node.ctx
        )


def eudf(expr, args, debug=False):
    assert len(args) > 0
    args = list(args.values())
    ndim = args[0].ndim

    tree = ast.parse(expr)                     # AST of the input expression
    tree = Transformer(ndim).visit(tree)       # AST of the UDF function
    source = decompile(tree)                   # Source code of the UDF function
    if debug:
        print(source)

    exec(source, globals(), locals())          # The Python function
    py_func = locals()['f']
    udf_func = udf.jit(py_func, source=source) # The UDF function
    ia_expr = ia.expr_from_udf(udf_func, args) # The IArray expression
    return ia_expr
