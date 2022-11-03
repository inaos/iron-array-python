# ironArray API and divergences with the array API standard

ironArray API tries to follow the [array_api](https://data-apis.org/array-api/latest/API_specification/index.html) standard, but there may be some divergences.

## Array object

### Operators

* From the arithmetic operators, `__floordiv__` and `__mod__` are not supported yet.
* The dtype of the result obtained in the comparison operators will be determined by the operands. If you would like to view the result as a specific dtype, you can set np_dtype of the result or when evaluating the lazy expression.
* None of the in-place (arithmetic, array and bitwise) operators are supported yet. The arithmetic operators **could be implemented**.
* From the reflected operators, `__rfloordiv__` and `__rmod__` are not supported yet.
* None of the bitwise operators are supported yet.
* `self` must have two dimensions in `__matmul__`.
* If arrays have more than 2 dimensions, `__matmul__` does not compute the multiplication from the last two dimensions.

All the supported operators return a lazy expression, except for `__matmul__`, `__rmatmul__` and `__pos__`, which return an array.

Broadcasting between arrays is not supported since the operands must have the same shape. The only exception to this rule is when one of the operands is a scalar, in which case it is promoted to the shape of the other array operand.

Type promotion rules do not follow the standard since the operands must have the same dtype (**could be implemented**). The only exception to this rule is when one of the operand is a scalar, in which case it is casted to the dtype of the other array operand.

The behaviour in special cases may also differ from the array API standard.


### Attributes

* `mT` is not supported yet.
* Maximum `ndim` value supported is 8. 

### Methods

* `__and__`, `__dlpack__`, `__dlpack_device__`, `__floordiv__`, `__invert__`, `__lshift__`, `__mod__`, `__or__`, `__rshift__`, `__xor__` and `to_device` are not supported yet. The rest of the methods from the array API not listed, and that must return an array, return instead a lazy expression which must be evaluated in order to obtain the result.

The behaviour of some methods regarding `Nan` and `infinity` values is implementation defined and may not follow the standard.

## Broadcasting

Broadcasting is only supported for scalars.

## Creation Functions

* `eye`, `from_dlpack`, `meshgrid`, `tril` and `triu`  are not supported yet.
* `arange` and `linspace` also support the `shape` param to build multi-dimensional arrays.
* `linspace` can also create arrays of numeric or boolean type.
* `asarray` can only avoid the copy when the array is an IArray instance.
* Since ironArray is CPU only, the only supported device is `"cpu"`.
* The default dtype is determined by the config defaults.

## Data Type Functions

* `broadcast_arrays`, `broadcast_to`, `can_cast`, `finfo`, `iinfo`, `result_type` are not implemented yet.
* Since the code is optimized for doing so, `astype`'s `copy` param is `False` by default and the copy can be avoided even when dtypes are different. Thus, it only performs a copy if `copy` is specified as `True`.

## Element-wise functions

* `acosh`, `asinh`, `atanh`, all the bitwise functions, `floor_divide`, `isfinite`, `isinf`, `isnan`, `log2`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `remainder`, `round`, `sign`, `trunc`, are not supported yet.
* The behaviour with special cases is implementation defined.
* Type promotion rules do not exist since the operands must have the same dtype. The only exception to this rule is when one of the operands is a scalar, in which case it is casted to the dtype of the other array operand.
* All these functions return a lazy expression except for `positive`.

## Linear Algebra Functions

* `tensordot` and `vecdot` are not implemented yet.
* Cannot compute transpose of a stack of matrices, only 2-dim arrays.

## Manipulation functions

None of them are supported yet. If you want to expand or shrink an array use `iarray.resize` function. Some of them **could be implemented**.

## Searching functions

None of them are supported yet as functions. `argmax`, `argmin` and maybe `nonzero` **could be implemented**. 

## Set functions

None are supported yet.

## Sorting functions

None are supported yet.

## Statistical functions

* The `keepdims` param will always be `False`. If it is set to `True` an error will be raised. **Could be implemented**

## Utility functions

`any` and `all` only accept bool dtype arrays. The parameter `keepdims` is not supported yet since ironArray never includes the reduced axes in the result.

## Function and methods signatures

* The functions/methods not listed in the standard may have their optional parameters as positional and keyword arguments.
