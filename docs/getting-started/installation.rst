Installation
============

In order to clone the repo with all the submodules, do::

    git clone --recurse-submodules https://github.com/inaos/iron-array-python


In general, you can update your local repo with::

    git pull
    git submodule update --recursive



Build
-----

In case you have Intel IPP libraries installed (for a turbo-enabled LZ4 codec within C-Blosc2),
make sure that you run::

    source ~/intel/bin/compilervars.sh intel64


so as to allow the iarray library to find the IPP libraries.

We rely on scikit-build, numpy and others to build and test the package, so please be sure to
install the requisites in your environment::

    python -m pip install -r requirements.txt


In addition, we need LLVM development and SVML packages that can be easily installed from conda::


    conda install -c intel mkl-include  # MKL
    conda install -c intel mkl-static  # MKL
    conda install -c intel icc_rt
    conda install -c numba llvmdev  # LLVM


We can proceed now with the compilation of the actual Python wrapper for iarray::

    rm -rf _skbuild iarray/iarray-c-develop/build/* iarray/*.so*    # *.pyd* if on windows (total cleanup and optional)
    python setup.py build_ext -j 4 --build-type=RelWithDebInfo  # choose Debug if you like


This will compile the iron-array C library and the Python extension in one go and will put both
libraries in the iarray/ directory, so the is wrapper is ready to be used right away.  As the
whole process is driven with cmake, making small changes in either the C library or the Python
extension will just trigger the re-compilation of the affected modules.

Also note the `-j 4` flag; this is a way to specify the number of processes in parallel that
you want to use during the build process.

Thanks to the nice integration of scikit-build with cmake, you can even pass [cmake configure
options directly from commandline](https://scikit-build.readthedocs.io/en/latest/usage.html#cmake-configure-options).
For example::

    python setup.py build_ext -j 4 --build-type=RelWithDebInfo -DDISABLE_LLVM_CONFIG=False


Test
++++

We can run the tests using::

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


Build wheels
++++++++++++

One can build wheels for the current platform with::

    python setup.py bdist_wheel


The wheels will appear in dist/ directory.

Note: see https://github.com/pypa/auditwheel package on how to audit and amend wheels for compatibility with a wide variety of Linux distributions.

Install
-------

You may want to install this package in the system.  For doing this, use::

    python setup.py install

