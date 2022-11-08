# Announcing ironArray 2022.3: high performance computing with multidimensional, compressed arrays

## What is new?

Besides the typical bug squashing work, the big thing here is that we have started to adopt the
new array [API standard](https://data-apis.org/array-api/latest/API_specification/index.html).
This includes the following changes:

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

There is still a lot to do before ironArray can be fully compliant with the array API standard,
but we think this is big first step forward.  Please test it and report any inconsistency or issue
you might detect.

Also, we are introducing a new `IArray.split()` method and its `concatenate()` counterpart for
efficiently splitting/concatenating IArrays in/from chunks.  Combine these with the new `from_cframe()`
and existing `IArray.to_cframe()` serialization method, and you will get an easy yet efficient way to send
data to your remote workers (like Spark clusters) and regenerate them on new `IArray` arrays.

For more information, see the release notes at: https://github.com/ironArray/ironArray-support/releases

## What is it?

ironArray is a C and Python library and a format that allows to compute with
large multidimensional, compressed datasets.  It takes every measure to consume as little resources as possible in order to cope with very large arrays without the need for using large computer facilities. 

Data can be manipulated either in-memory or on-disk, using the same API. By applying artificial intelligence, the integrated compute engine can decide in real time which computing method and compression strategy is best depending on the needs of the user (best speed, best compression ratio or a balance among the two).

For more info, see: https://ironarray.io
