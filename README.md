# iron-array-python
IronArray for Python

## Build modes

This package supports two modes for building the Python extension: develop and release.

* The develop mode uses the `iarray` repo locallly so as to find the headers and libraries (you must ensure that the libraries can be found in the `build/` directory of the local repo).  This allows to develop both packages (the C library and the Python wrapper) in parallel without the need to re-install the C library in every iteration.

* The release mode assumes that both the `iarray` library is installed in the system.

The two modes are setup and driven by environment variables.  Here are examples:

### Develop mode

```
$ export IARRAY_DEVELOP_MODE=True
$ export INAC_DIR=~/.inaos/cmake/inac-darwin-x86_64-relwithdebinfo-1.0.4
$ export IARRAY_DIR=../iron-array  # the default; if your path is this one, no need to set this
$ export PYTHONPATH=.
```

### Release mode

```
$ unset IARRAY_DEVELOP_MODE
$ export INAC_DIR=$HOME/.inaos/cmake/inac-darwin-x86_64-relwithdebinfo-1.0.4
$ export PYTHONPATH=.
```

## Compile

After setting up the build mode, we can proceed with the compilation of the actual Python wrapper for iarray:

```
$ rm -rf build iarray/*.so    # *.pyd if on windows.  This step is a cleanup and purely optional.
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

## Install

When in release mode, you may want to install this package in the system.  For doing this, use:

```
$ python setup.py install
```

The setup.py can be used to produce wheels, where all the libraries are included (see https://pythonwheels.com).
