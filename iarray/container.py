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


class LazyIArray:

    def __init__(self, ctx, new_op):
        value1, op, value2 = new_op
        self.ctx = ctx
        self.shape = value1.shape
        self.pshape = value1.pshape
        print("shape ->", self.shape, self.pshape)
        self.expression = f"(o0 {op} o1)"
        self.operands = {"o0": value1, "o1": value2}


    def eval(self):
        out = ext.empty(self.ctx, shape=self.shape, pshape=self.pshape)
        block_size = self.pshape
        operand_iters = tuple(o.iter_block(block_size) for o in self.operands.values())
        all_iters = operand_iters + (out.iter_write(),)   # put the iterator for the output at the end
        for block in zip(*all_iters):
            block_operands = {o: block[i][1] for (i, o) in enumerate(self.operands.keys())}
            out_block = block[-1][1]  # the block for output is at the end, by construction
            out_block[:] = ne.evaluate(self.expression, local_dict=block_operands)
        del all_iters  # TODO: fix the iterators so that we don't have to do this manually to make them go
        print("despres d'esborrar els iteradors...")
        return IArray(self.ctx, c=out.to_capsule())


    def __str__(self):
        expression = f"{self.expression}\n"
        return expression


class IArray(ext.Container):

    def __init__(self, ctx=None, c=None):
        if ctx is None:
            # Assign a default context
            cfg = ext.Config()
            ctx = ext.Context(cfg)
        self.ctx = ctx
        if c is None:
            raise ValueError("You must pass a Capsule to the C container struct in the IArray constructor")
        super(IArray, self).__init__(ctx, c)


    def __add__(self, value):
        return LazyIArray(self.ctx, new_op=(self, '+', value))


    def __sub__(self, value):
        return LazyIArray(self.ctx, new_op=(self, '-', value))


    def __mul__(self, value):
        return LazyIArray(self.ctx, new_op=(self, '*', value))


    def __truediv__(self, value):
        return LazyIArray(self.ctx, new_op=(self, '+', value))



if __name__ == "__main__":
    # Create iarray context
    cfg = ext.Config()
    ctx_ = ext.Context(cfg)

    # Define array params
    shape = [100]
    pshape = [20]
    size = int(np.prod(shape))

    # Create initial containers
    a1 = ext.linspace(ctx_, size, 0, 10, shape, pshape, "double")
    a2 = ext.linspace(ctx_, size, 0, 20, shape, pshape, "double")
    a3 = a1 + a2
    print(a3)
    a4 = a3.eval()
    a4_np = ext.iarray2numpy(a4.ctx, a4)
    print(a4_np)
