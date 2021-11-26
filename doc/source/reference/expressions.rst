-----------
Expressions
-----------

ironArray has a strong support for expression evaluation. Things like sums,
products, divisions and a pretty complete range of transcendental functions
(e.g. exp, sin, asin, tanh…) have been implemented so as to guarantee an
efficient evaluation in (large) arrays.

Expressions can be built either from small one liners (either in string format
or as regular Python expressions, see tutorials section for details), or from
User Defined Functions which are described later in section `UDFs`_.

Expressions are implemented in the :py:class:`iarray.Expr` class.  Once built,
they can be evaluated via the :func:`iarray.Expr.eval` method.

.. currentmodule:: iarray

Expr class
==========

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   Expr

Constructors
------------

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   expr_from_string
   expr_from_udf


Methods
-------

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   Expr.eval


.. _UDFs:

User Defined Functions
=======================

User Defined Functions (or UDFs for short) are small computational kernels for
data in arrays that can be expressed in a simple subset of Python. These
functions are then passed to the internal compiler in ironArray and a binary
specific and optimized for the local CPU is generated. In addition, it will
make use of the available SIMD hardware in the CPU for accelerating
transcendental functions and other operations.

Overview
--------

An UDF receives windows (blocks) of data from different arrays and after some
computations, it generates back another data block for being stored in a window
of the output array.  The `out` window is always the first parameter and the
different inputs come later on.

The UDF uses Python annotations so as to express properties like the data types
and the number of dimensions of the arguments.

The function *must* return an integer, 0 for success and non-zero for error.
The type of the return value can be annotated, but it's not necessary as it
defaults to integer (which is the only supported type).

Here it is a simple example of how this works in practice::

    from iarray.udf import jit, Array, float32

    @jit
    def mean_with_cap(out: Array(float32, 3),
                      p1: Array(float32, 3),
                      p2: Array(float32, 3),
                      p3: Array(float32, 3)):

        l = p1.window_shape[0]
        m = p1.window_shape[1]
        n = p1.window_shape[2]

        MAX_VAL = 1e10  # this cap cannot be exceed when doing the mean
        for i in range(l):
            for j in range(m):
                for k in range(n):
                    value = p1[i,j,k] + p2[i,j,k] + p3[i,j,k]
                    value = value if value < MAX_VAL else 0
                    out[i,j,k] = value / 3

        return 0

    precip_expr = ia.expr_from_udf(mean_with_cap, [precip1, precip2, precip3])
    precip_mean = precip_expr.eval()

Above we have an UDF called `mean_with_cap` that produces an `out` data block
of `float32` with 3 dimensions, and has 3 inputs (`p1`, `p2`, `p3`) also with a
`float32` type and 3 dimensions as well.  The function returns an `int` as an
error code (0 means `success`).  Then, it retrieves the shape of the view of
one of the inputs (ironArray guarantees that all the views in inputs and output
have the same shape) and does a nested `for loop` for computing the mean for
every value in the inputs.

After that, the UDF is compiled and produces an :class:`Expr` expression via
the :func:`expr_from_udf` call.  The inputs are passed in the
:func:`expr_from_udf` call too, but there is no need to create the `out` array
as one will be properly crafted and passed to the UDF during the execution of
the `precip_expr.eval()` call.  The different `out` blocks coming out of the
UDF are assembled internally by the computational engine in ironArray and
delivered in the final `precip_mean` array.

See the tutorial section for more hints on how to use UDFs.

Types and operations
-----------------------------------------

The supported numerical types are:

- floats, `float32` and `float64`
- integers, `int8`, `int32` and `int64`

This includes literals. If not annotated an integer literal will be interpreted
as `int64`, and a float literal will be interpreted as `float64`. For example::

    n = 5          # int64
    n:int8 = 5     # int8
    n = 5.0        # float64
    n:float32 = 5  # float32

The supported binary operations are addition `+`, substraction `-`,
multiplication `*`, and division `/`. The supported unary operators for
numerical types are `-`.

The modulus operator `%` is supported as well, but only if at least one of the
operands is a variable (i.e. ``i % 3`` works but ``7 % 3`` does not).

In numerical expressions mixing an integer and a float, the integer value will
be coerced to float. If mixing values with different size, the smaller one will
be coerced to the larger one (e.g. int32 and int64, the int32 value will be
coerced to int64).

Comparisons are also supported, with the operators for: equality `=`, not
equality `!=`, less than `<`, less than equal `<=`, greater than `>`, greater
than equal `>=`. And logical expressions with the `and`, `or` and `not`
operators. But boolean literals are not supported.

Finally, the conditional expression `x if condition else y` is supported.

Supported Python statements and functions
-----------------------------------------

The current supported statements are:

- assignmets, including augmented and annotated assignments
- loops, `for` and `while`
- conditionals, `if/elif/else`

The syntax is exactly the same as in Python itself (actually, UDFs are parsed
using the same AST parser than Python).

Also, the `range(stop)` function is supported. The optional arguments start and
step are not supported.

Supported math operations
-------------------------

All the basic arithmetic operations (+, -, \*, /) are supported.  In addition,
the whole set of mathematical functions in ironArray expressions are supported
too (see :ref:`Math Functions`).

Data window metainfo
--------------------

When doing operations within blocks of data, sometimes it is interesting to
know the absolute position of the current data windows for the different
arrays.  You can achieve this through different properties:

window_shape
~~~~~~~~~~~~

The shape of the current window for the array.

window_start
~~~~~~~~~~~~

The starting coordinates in each dimension for the current window for the
array.

window_strides
~~~~~~~~~~~~~~

Number of elements to step in each dimension when traversing the window.
