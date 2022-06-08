.. _UDFs:

----------------------
User Defined Functions
----------------------

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

The supported numerical types are floats (`float32` and `float64`).

This includes literals. If not annotated a float literal will be interpreted as
`float64`. For example::

    n = 5.0        # float64
    n:float32 = 5  # float32

The supported binary operations are addition `+`, subtraction `-`,
multiplication `*`, and division `/`. The supported unary operators for
numerical types are `-`.

The modulus operator `%` is supported as well, but only if at least one of the
operands is a variable (i.e. ``i % 3`` works but ``7 % 3`` does not).

In numerical expressions mixing a float32 with a float64, the float32 value
will be converted to float64.

Comparisons are also supported, with the operators for: equality `=`, not
equality `!=`, less than `<`, less than equal `<=`, greater than `>`, greater
than equal `>=`. And logical expressions with the `and`, `or` and `not`
operators. But boolean literals are not supported.

Finally, the conditional expression `x if condition else y` is supported also.

Supported Python statements and functions
-----------------------------------------

The current supported statements are:

- assignments, including augmented and annotated assignments (currently integers
  are automatically cast to floats, whereas booleans raise an error)
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
as well (see :ref:`Math Functions`).

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

Number of elements to step in each dimension for reaching the next element when traversing the window.

Libraries of scalar UDFs
------------------------

Scalar UDFs are a special case of UDFs that only accept scalars as parameters instead of arrays. The advantage of scalar UDFs
is that they can be called either from expressions or from regular UDFs (also known as vector UDFs). These scalar UDFs
needs to be registered in libraries in order to be used.

For instance, we can create a scalar UDF like the following::

    @udf.scalar(lib="lib")
    def fsum(a: udf.float64, b: udf.float64) -> float:
        if a < 0:
            return -a + b
        else:
            return a + b

This function has been registered to the UDFRegistry "lib", which has been automatically created.

To use this function inside an expression, you need to specify the library in the following way::

    expr = "4 * lib.fsum(x, y)"
    expr = ia.expr_from_string(expr, {"x": x, "y": 1})
    z = expr.eval()

You can also use them inside a regular UDF::

    @udf.jit
    def udf_sum(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1), y: udf.float64):
        for i in range(out.shape[0]):
            out[i] = 4 * lib.fsum(x[i], y)
        return 0

    expr2 = ia.expr_from_udf(udf_sum, [x], [1])
    z = expr2.eval()

For more info on how to deal with the libraries registry see :ref:`UdfRegistry`.
