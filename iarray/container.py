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
import numexpr as ne
import iarray.iarray_ext as ext
import iarray as ia


def rebase_operands(operands, new_base):
    new_operands = {}
    for old_op in operands.keys():
        old_pos = int(old_op[1:])
        new_pos = old_pos + new_base
        new_op = f"o{new_pos}"
        new_operands[new_op] = operands[old_op]
    return new_operands


def rebase_expression(expr, new_base):
    new_expr = ""
    skip_to_char = 0
    for i in range(len(expr)):
        if i < skip_to_char:
            continue
        if expr[i] == 'o':
            new_expr += 'o'
            try:
                j = expr[i+1:].index(' ')
            except ValueError:
                j = expr[i + 1:].index(')')
            old_pos = int(expr[i+1:i+j+1])
            new_pos = old_pos + new_base
            new_expr += f"{new_pos}"
            skip_to_char = i + j + 1
        else:
            new_expr += expr[i]
    return new_expr


class LazyExpr:

    def __init__(self, new_op, ctx=None):
        # This is the very first time that a LazyExpr is formed from two operands that are not LazyExpr themselves
        value1, op, value2 = new_op
        self.ctx = ctx
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            self.expression = f"({value1} {op} {value2})"
        elif isinstance(value2, (int, float)):
            self.operands = {"o0": value1}
            self.expression = f"(o0 {op} {value2})"
        elif isinstance(value1, (int, float)):
            self.operands = {"o0": value2}
            self.expression = f"({value1} {op} o0)"
        else:
            self.operands = {"o0": value1, "o1": value2}
            self.expression = f"(o0 {op} o1)"

    def update_expr(self, new_op):
        # One of the two operands are LazyExpr instances
        value1, op, value2 = new_op
        if isinstance(value1, LazyExpr) and isinstance(value2, LazyExpr):
            # Expression fusion
            total_ops1 = len(value1.operands)
            # Take expression 2 and rebase the operands
            new_expr = rebase_expression(value2.expression, total_ops1)
            self.expression = f"({self.expression} {op} {new_expr})"
            self.operands.update(rebase_operands(value2.operands, total_ops1))
        elif isinstance(value1, LazyExpr):
            if isinstance(value2, (int, float)):
                self.expression = f"({self.expression} {op} {value2})"
            else:
                op_name = f"o{len(self.operands)}"
                self.operands[op_name] = value2
                self.expression = f"({self.expression} {op} {op_name})"
        else:
            if isinstance(value1, (int, float)):
                self.expression = f"({value1} {op} {self.expression})"
            else:
                op_name = f"o{len(self.operands)}"
                self.operands[op_name] = value1
                self.expression = f"(o{op_name} {op} {self.expression})"
        return self


    def __add__(self, value):
        return self.update_expr(new_op=(self, '+', value))

    def __radd__(self, value):
        return self.update_expr(new_op=(value, '+', self))

    def __sub__(self, value):
        return self.update_expr(new_op=(self, '-', value))

    def __rsub__(self, value):
        return self.update_expr(new_op=(value, '-', self))

    def __mul__(self, value):
        return self.update_expr(new_op=(self, '*', value))

    def __rmul__(self, value):
        return self.update_expr(new_op=(value, '*', self))

    def __truediv__(self, value):
        return self.update_expr(new_op=(self, '/', value))

    def __rtruediv__(self, value):
        return self.update_expr(new_op=(value, '/', self))


    def eval(self):
        # TODO: see if ctx, shape and pshape can be instance variables, or better stay like this
        o0 = self.operands['o0']
        if self.ctx is None:
            # Choose the context of the first operand
            self.ctx = o0.ctx
        shape_ = o0.shape
        pshape_ = o0.pshape
        out = ia.empty(self.ctx, shape=shape_, pshape=pshape_)
        operand_iters = tuple(o.iter_block(pshape_) for o in self.operands.values() if isinstance(o, IArray))
        all_iters = operand_iters + (out.iter_write(),)   # put the iterator for the output at the end
        # all_iters =  (out.iter_write(),) + operand_iters  # put the iterator for the output at the front
        for block in zip(*all_iters):
            block_operands = {o: block[i][1] for (i, o) in enumerate(self.operands.keys(), start=0)}
            out_block = block[-1][1]  # the block for output is at the end, by construction
            # block_operands = {o: block[i][1] for (i, o) in enumerate(self.operands.keys(), start=1)}
            # out_block = block[0][1]  # the block for output is at the front, by construction
            ne.evaluate(self.expression, local_dict=block_operands, out=out_block)
        return out


    def __str__(self):
        expression = f"{self.expression}"
        return expression


class IArray(ext.Container):

    def __init__(self, ctx=None, c=None):
        if ctx is None:
            # Assign a default context
            cfg = ia.Config()
            ctx = ia.Context(cfg)
        self.ctx = ctx
        if c is None:
            raise ValueError("You must pass a Capsule to the C container struct in the IArray constructor")
        super(IArray, self).__init__(ctx, c)


    def __add__(self, value):
        return LazyExpr(new_op=(self, '+', value))

    def __radd__(self, value):
        return LazyExpr(new_op=(value, '+', self))

    def __sub__(self, value):
        return LazyExpr(new_op=(self, '-', value))

    def __rsub__(self, value):
        return LazyExpr(new_op=(value, '-', self))

    def __mul__(self, value):
        return LazyExpr(new_op=(self, '*', value))

    def __rmul__(self, value):
        return LazyExpr(new_op=(value, '*', self))

    def __truediv__(self, value):
        return LazyExpr(new_op=(self, '/', value))

    def __rtruediv__(self, value):
        return LazyExpr(new_op=(value, '/', self))


if __name__ == "__main__":
    # Create iarray context
    cfg_ = ia.Config()
    ctx_ = ia.Context(cfg_)

    # Define array params
    shape = [40]
    pshape = [20]
    size = int(np.prod(shape))

    # Create initial containers
    a1 = ia.linspace(ctx_, size, 0, 10, shape, pshape, "double")
    a2 = ia.linspace(ctx_, size, 0, 20, shape, pshape, "double")
    a3 = a1 + a2 + a1 - 2 * a1 + 1
    print(a3)
    a3 += 2
    print(a3)
    a4 = a3.eval()
    a4_np = ia.iarray2numpy(a4.ctx, a4)
    print(a4_np)
