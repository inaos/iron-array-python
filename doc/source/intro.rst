------------
Introduction
------------

ironArray for Python at glance
==============================

ironArray is a C library oriented to compute and handle large multidimensional arrays efficiently.
ironArray for Python is the official wrapper to it.

Among its main features, we can list:

* Multidimensional and arbitrarily large arrays of floating point data (float32 and float64).

* Transparent, high-performance compression. The arrays are compressed and decompressed internally, without the need for user intervention. Use memory and disk resources more efficiently.

* Advanced compute engine, based on LLVM, for evaluating expressions with arrays as operands, reductions and a subset of linear algebra.

* Pervasive parallelism. All the operations are performed using a tuned number (see below) of the cores in the system.

* Native persistency. The arrays can be stored on disk and loaded very efficiently. Arrays can even be operated with without the need to load them first in-memory (out-of-core operation).

* Automatic fine-tuning based on underlying hardware. Important parameters of your CPU, like the number of cores or cache sizes, are automatically detected and used for optimal execution times.

* Python and C APIs. ironArray is a pure C library that comes with an easy to use Python wrapper. Use whatever you prefer.

* Easy to install. ironArray comes with Python wheels (which include the headers and binary libraries of the C library) for a fast deployment in all major platforms (Linux, Windows, Mac).


Basic operations
================

TODO: one should use images so as to better illustrate the different operations.

Array creation
--------------

So as to better feel how the interface looks like...

Array computations
------------------


Reductions
----------

Linear Algebra
--------------
