import ast


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

    def visit_Module(self, node):
        self.generic_visit(node)
        return ast.Module(body=[
            ast.FunctionDef(
                name='f',
                args=ast.arguments(
                    posonlyargs=[],
                    args=[
                        ast.arg('out', annotation=self.annotation),
                        ast.arg('x', annotation=self.annotation),
                        ast.arg('y', annotation=self.annotation),
                    ],
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
        return ast.Subscript(
            value=name(node.id),
            slice=ast.Index(value=name(self.index)),
            ctx=node.ctx
        )


def eudf(expr, args):
    assert len(args) > 0
    arg = list(args.values())[0]
    ndim = arg.ndim

    tree = ast.parse(expr)
    return Transformer(ndim).visit(tree)
