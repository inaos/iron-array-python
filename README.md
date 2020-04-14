# iron-array-python
IronArray for Python

## Clone repository and iron-array C library

First, clone the Python wrapper repo with:

```bash
git clone git@github.com:/inaos/iron-array-python
```

Now, clone the C library repo in the iarray/ directory:

```bash
cd iron-array-python/iarray
git clone git@github.com:/inaos/iron-array
cd iron-array
git submodule update --init
cd ../..
```

### Build

For linking with Intel libraries (MKL, IPP...), make sure that you run:

```bash
source ~/intel/bin/compilervars.sh intel64
```

We rely on scikit-build to build the package, so please be sure that this is installed on your environment:

```bash
conda install -c conda-forge scikit-build
```

We can proceed now with the compilation of the actual Python wrapper for iarray:

```bash
rm -rf _skbuild iarray/iron-array/build  # this step is a cmake cache cleanup and purely optional
python setup.py build_ext -j 4   # build with RelWithDebInfo by default
# python setup.py build_ext -j 4 --build-type=Debug  # enforce the Debug mode
```

This will compile the iron-array C library and the Python extension in one go and will put both libraries in the iarray/ directory, so the wrapper is ready to go.  As the whole process is cmake driven, making small changes in either the C library or the Python extension will just trigger the re-compilation of the affected modules, making the re-compilation process pretty fast.

Also note the `-j 4` flag; this is a way to specify the number of processes in parallel that you want to use during the build process; adjust it to your own preferences.

You can even pass [cmake configure options directly from commandline](https://scikit-build.readthedocs.io/en/latest/usage.html#cmake-configure-options).

### Test

```bash
pytest                                                                                           (base)
=================================================================== test session starts ====================================================================
platform darwin -- Python 3.7.4, pytest-4.3.0, py-1.8.1, pluggy-0.13.1
hypothesis profile 'default' -> database=DirectoryBasedExampleDatabase('/Users/faltet/inaos/iron-array-python/.hypothesis/examples')
rootdir: /Users/faltet/inaos/iron-array-python, inifile:
plugins: hypothesis-4.18.3
collected 164 items

iarray/tests/test_constructor.py ............................                                                                                        [ 17%]
iarray/tests/test_copy.py .......                                                                                                                    [ 21%]
iarray/tests/test_expression.py .........................................................                                                            [ 56%]
iarray/tests/test_iterator.py ........                                                                                                               [ 60%]
iarray/tests/test_load_save.py ....                                                                                                                  [ 63%]
iarray/tests/test_matmul.py ................                                                                                                         [ 73%]
iarray/tests/test_random.py ........................................                                                                                 [ 97%]
iarray/tests/test_slice.py ....                                                                                                                      [100%]

========================================================== 164 passed, 0 warnings in 7.13 seconds ==========================================================```
```

## Build wheels

One can build wheels for the current platform with 

```bash
python setup.py bdist_wheel
```

The wheels will appear in dist/ directory.

Note: see https://github.com/pypa/auditwheel package on how to audit and amend wheels for compatibility with a wide variety of Linux distributions.

## Install

You may want to install this package in the system.  For doing this, use:

```bash
python setup.py install
```
