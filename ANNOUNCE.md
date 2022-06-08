# Announcing ironArray 2022.2: high performance computing with multidimensional, compressed arrays

## What is new?

This is a much improved version providing many new features like:

* Calling UDFs inside expressions,
* Enhanced masks in expressions
* IArray objects can be expanded/shrinked in any dimension *and* in any desired position (i.e. insertions are supported :-)
* A new proxy for Zarr arrays that can be used as a native IArray array.
* Multi-threaded constructors (e.g. random arrays can be built up to 10x faster than NumPy)
* Arrays can be viewed as another data type (equivalent to `.astype()` in NumPy).
* New `ia.std()` and `ia.median()` reductions.  This mimics the same functions in NumPy.
* Support for ignoring NaN values in reductions via `ia.nanmean()` and friends.

For more information, see the release notes at: https://github.com/ironArray/ironArray-support/releases

## What is it?

ironArray is a C and Python library and a format that allows to compute with
large multidimensional, compressed datasets.  It takes every measure to consume as little resources as possible in order to cope with very large arrays without the need for using large computer facilities. 

Data can be manipulated either in-memory or on-disk, using the same API. By applying artificial intelligence, the integrated compute engine can decide in realtime which computing method and compression strategy is best depending on the needs of the user (best speed, best compression ratio or a balance among the two).

For more info, see: https://ironarray.io
