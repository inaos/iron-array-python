Release notes for ironArray 2022.2
==================================

Changes from 2022.2 to 2022.3
=============================

* Adopted array [API standard](https://data-apis.org/array-api/latest/API_specification/index.html) (when possible), this includes the following changes:
  * `astype` has been changed from a method to a function.
  * `transpose` function was renamed to `matrix_transpose`.
  * `size` has been added as an attribute of `IArray`.
  * Mathematical functions cannot be used as methods anymore.
  * The trigonometric functions beginning with `arc*` can no longer be used; use `a*` instead.
  * `absolute` has been renamed to `abs`.
  * `negate` has been renamed to `negative`.
  * `power` has been renamed to `pow`.
  * `empty_like`, `ones_like`, `full_like`, `zeros_like`, and `asarray` constructors have been added.
  * `arange`: param `shape` is left as a keyword-only argument and `step` param is 1 by default.
  * `linspace`: param `shape` is left as a keyword-only argument and a `num` param has been added. 
  * Before, all parameters could be positional or keyword parameters; now some parameters are positional-only and some others are keyword-only.
  * In `std`, `var`, `nanstd` and `nanvar` you can now choose the degrees of freedom adjustment with the `correction` keyword-only parameter.
  * `all` and `any` reductions have been added for all the supported types.
  * Type restrictions have been added to some functions and methods. For example, the trigonometric functions will only work with floating-point data arrays.
  * `add`, `divide`, `expm1`, `greater`, `greater_equal`, `less`, `less_equal`, `log1p`, `logaddexp`, `multiply`, `not_equal`, `positive`, `square` and `subtract` functions have been added.
  * `__pos__`, `__neg__`, `__pow__`, `__bool__`, `__float__`, `__int__` methods have been added.
  * When a function used to return a Python scalar, it returns now a 0-dim IArray.

* Fixed a crash when evaluating expressions with zarr proxies.

* Reduction functions can take a new `oneshot` bool argument.  Oneshot algorithm normally uses less memory, albeit is slower in general. Default is False.

* New `concatenate()` function for concatenating a list of one-chunk arrays into one with a specified shape.

* New `from_cframe()` function for serializing an array into a cframe (`bytes` object).  This is useful to e.g. sending the data to workers. Complements the `IArray.to_cframe()` method.

* New `IArray.split()` method for efficiently splitting an array into a list of one-chunk arrays.  You can use `concatenate()` function to assemble the parts later on.

* New `IArray.slice_chunk_index()` method for getting slices on an array using plain chunk indices.


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

* IArray objects can be expanded/shrinked in any dimension *and* in any desired position.  See the new docstrings for `IArray.resize`, `IArray.insert`, `IArray.append` and `IArray.delete`.

* Support for lossy ZFP.  This has been inherited from its recent addition to the Blosc2 compressor.  For more info see: https://www.blosc.org/posts/support-lossy-zfp/

* New `ia.zarr_proxy` constructor to wrap a zarr array and make it work as it is a native `IArray` object.

* Added support for the `**` operator (meaning `power()`) to expressions.  This is to mimic the Python operators.

* Multi-threaded constructors.  Now constructors work in multi-threading mode by default, meaning that they are much faster now.  This specially improves the speed of the random constructors (up to 10x faster than the equivalent in Numpy, as measured on a machine with 16 cores).

* Support for NumPy dtypes. Now you can set the `np_dtype` parameter in constructors to any NumPy dtype that you want. This will be used when data is converted from an IArray container to NumPy. The only missing NumPy dtypes are strings, complex and structured dtypes.

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
