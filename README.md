# iron-array-python
IronArray for Python

## Compile

For now, this package is setup so that an IronArray repository is installed locally, and both the iarray and blosc libraries have been compiled in their respective build directories.  With that, you must setup the next environment variables:

```
$ export INAC_DIR=$HOME/.inaos/cmake/inac-darwin-x86_64-relwithdebinfo-1.0.4
$ export IARRAY_DIR=../iron-array
$ export BLOSC_DIR=$IARRAY_DIR/contribs/c-blosc2
$ export PYTHONPATH=.
```

We can now proceed with the compilation of the actual Python wrapper for iarray with:

```
$ python setup.py build_ext -i
```

and  execute the tests with:

```
$ pytest
====================================================================================== test session starts =======================================================================================
platform darwin -- Python 3.7.1, pytest-4.3.0, py-1.8.0, pluggy-0.9.0
rootdir: /Users/faltet/inaos/iron-array-python, inifile:
collected 16 items

iarray/tests/test_constructor.py ..............                                                                                                                                            [ 87%]
iarray/tests/test_expression.py ..                                                                                                                                                         [100%]

=================================================================================== 16 passed in 0.29 seconds ====================================================================================
```

That's all folks!
