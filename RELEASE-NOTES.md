Release notes for ironArray 2022.2
==================================

Changes from 2022.1 to 2022.2
=============================

* UDFs can be registered and used inside regular expressions.  Here it is an example:

```
@udf.scalar(lib="lib")
def fsum(x: udf.float64, y: udf.float64) -> float:
    return x + y

expr = "4 * lib.sum(x, y)"
expr = ia.expr_from_string(expr, {"x": a1, "y": 1})
b1 = expr.eval()
```

* Enhanced mask support for expressions.  Now you can evaluate things like: `x[((y == 3) & (z == 4)) | ~(x == 0)]` which in NumPy are equivalent to:

```
np.where(((y == 3) & (z == 4)) | ~(x == 0), x, np.nan)
```

* Slices and reductions can appear anywhere in the expression (e.g. `(x.min(axis=1) - 1.35) *  y[:,1]`, where `x` and `y` are 2-dim arrays).

* Lazy expressions gained the same functionality than recent enhancements in expressions (actually, expressions are now evaluated as lazy expressions).

* IArray objects can be expanded/shrinked in any dimension *and* in any desired position.  See the new docstrings for `IArray.resize`, `IArray.insert`, `IArray.append` and `IArray.remove`.

* Support for lossy ZFP.  This has been inherited from its recent addition to the Blosc2 compressor.  For more info see: https://www.blosc.org/posts/support-lossy-zfp/

* New `ia.zarr_proxy` constructor to wrap a zarr array and make it work as it is a native `IArray` object.

* Added support for the `**` operator (meaning `power()`) to expressions.  This is to mimic the Python operators.

* Multi-threaded constructors.  Now constructors work in multi-threading mode by default, meaning that they are much faster now.  This specially improves the speed of the random constructors (up to 10x faster than the equivalent in Numpy, as measured on a machine with 16 cores).

* Views of promoted types.  Now arrays with any type can be viewed as any other type that has a better representation range (e.g. float32 -> float64).  A view means that no copy is done, and that values are converted on the flight.

* Views of views are supported now.  This means that you can chain an unlimited number of views on a 'real' array (e.g. `a[1:4].astype(np.float64)`).

* New `ia.std()` and `ia.median()` reductions.  This mimics the same functions in NumPy.

* Support for ignoring NaN values in reductions via `ia.nanmean()` and friends.  This mimics the functionality in NumPy.


2022.1 (Initial Release)
========================

Initial support for:

* Compressed and N-dim data containers (IArray)
* Support for float32, float64 and int8, int16, int32, int64, with its unsigned versions
* Both in-memory and on-disk storage
* High quality random generators
* Vectorized evaluation of complex mathematical expressions
* Reductions (sum, prod, mean, min, max...)
* User Defined Functions that can handle IArrays
* Matrix multiplication and transposition