Release notes for ironArray 2022.2
==================================

Changes from 2022.1 to 2022.2
=============================

* UDFs can be registered and be used inside regular expressions.  Here it is an example of how it works:

```
@udf.scalar(lib="lib")
def fsum(x: udf.float64, y: udf.float64) -> float:
    return x + y

expr = "4 * lib.sum(x, y)"
expr = ia.expr_from_string(expr, {"x": a1, "y": 1})
b1 = expr.eval()
```

* New `ia.expr_get_operands` and `ia.expr_get_ops_funcs` function for getting the operands and the functions in a string expression.

* IArray objects can be expanded/shrinked in any dimension *and* in any desired position.  See the new docstrings for `IArray.resize`, `IArray.insert`, `IArray.append` and `IArray.remove`.

* Support for lossy ZFP is here!  This has been inherited from its recent addition to the Blosc2 compressor.  For more info see: https://www.blosc.org/posts/support-lossy-zfp/

* New `ia.zarr_proxy` constructor to wrap a zarr array and make it work as it is a native `IArray` object.

* Added support for the `**` operator (meaning `power()`) to expressions.  This is to mimic the Python operators.

* Multi-threaded constructors.  Now constructors work in multi-threading mode by default, meaning that they are much faster now.  This specially improves the speed of the random constructors (up to 10x faster than the equivalent in Numpy, provided a machine with a good number of cores indeed :-).
