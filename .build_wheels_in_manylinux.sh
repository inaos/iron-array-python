#!/bin/bash

set -x
#set -e  # exit when any command fails


mkdir -p $HOME/.inaos/cmake
mkdir -p $HOME/INAOS
echo "INAC_REPOSITORY_LOCAL=$HOME/INAOS" > $HOME/.inaos/cmake/repository.txt
echo "INAC_REPOSITORY_REMOTE=https://inaos.jfrog.io/inaos/libs-release-local/inaos" >> $HOME/.inaos/cmake/repository.txt
echo "INAC_REPOSITORY_USRPWD=licensed:AKCp5bBraH7CasbsYCURsjzkbjXwVwdYcT7u39EiuL6GjnK1VKfKQWCd1E2E64mHokU5YUHku" >> $HOME/.inaos/cmake/repository.txt

# Install pypirc
echo "[distutils]" > $HOME/.pypirc
echo "index-servers = jfrog" >> $HOME/.pypirc
echo "[jfrog]" >> $HOME/.pypirc
echo "repository: https://inaos.jfrog.io/artifactory/api/pypi/pypi-iron-array" >> $HOME/.pypirc
echo "username: iarray-deploy" >> $HOME/.pypirc
echo "password: M{wf5-Am^2)c8&vR" >> $HOME/.pypirc

# Activate the conda environment in this docker image
# This is mainly to install llvmdev and intel packages which are python-agnostic
source dockcross/install-conda.sh
source /work/conda/etc/profile.d/conda.sh
conda activate
export CONDA_PREFIX=/work/conda  # some old conda's do not set this

# Compile iron-array, the C library
pushd iarray/iarray-c-develop/
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDISABLE_LLVM_CONFIG=True -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
export MAKEFLAGS=-j$(($(grep -c ^processor /proc/cpuinfo) - 0))
cmake --build . --target iarray # is there a way to select just the static or the shared lib?
popd

# Remove previous existing wheels (for lacal testing mainly)
rm -rf dist

# Inform setup.py that we want to build wheels here
touch BUILD_WHEELS

########### Python specific work begins here ###########################
versions=(cp36-cp36m cp37-cp37m cp38-cp38)
#versions=(cp37-cp37m)

for version in "${versions[@]}"; do
  /opt/python/${version}/bin/python -m pip install --upgrade pip
  /opt/python/${version}/bin/python -m pip install cython numpy
  rm -rf _skbuild/
  /opt/python/${version}/bin/python setup.py build --build-type RelWithDebInfo -- -DDISABLE_LLVM_CONFIG=True -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
  # Copy the necessary shared libraries
  /bin/cp -f iarray/iarray-c-develop/build/libiarray.so iarray/
  # We need manylinux2014_x86_64 because icc_rt needs this:
  # OSError: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by /work/conda/lib/libintlc.so.5)
  # (manylinux2010 requires GLIB_2.12 or earlier: https://www.python.org/dev/peps/pep-0571/)
  /opt/python/${version}/bin/python setup.py bdist_wheel --plat-name manylinux2014_x86_64 -- -DDISABLE_LLVM_CONFIG=True -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:iarray
for whl in dist/*linux*.whl; do
  /opt/python/cp37-cp37m/bin/auditwheel show ${whl}
  # Looks like repairing cannot be done because the dependencies are not included,
  # and auditwheel is not able to find libintlc.so.5 which is in the icc-rt package (installed).
  # Error was: ValueError: Cannot repair wheel, because required library "libintlc.so.5" could not be located
  # Another possibility is to add it manually in setup.py, but still, not all the dependencies are nailed down.
  # Ok, so let's disable repairing for now (as another benefit, this leads to lighter wheels).
  # For the time being, lets add an install dependency on icc-rt for wheels.  This seems to work well.
  #/opt/python/cp37-cp37m/bin/auditwheel repair ${whl} -w /work/dist/  --plat manylinux2014_x86_64
done

# Test the installation of the wheel on a different conda environment
# (not very complete because some binaries may remain, but anyway)
# TODO: make this work
#for version in "${versions[@]}"; do
#  pybin=/opt/python/${version}/bin/python
#  python_version=`${pybin} -c "import sys; print('%d.%d'%sys.version_info[0:2])"`
#  conda create --yes -n test-wheels python=$python_version
#  conda activate test-wheels
#  cd /tmp/
#  python -m pip install iarray --user --no-cache-dir --no-index -f /work/dist/
#  cd /work/
#  python -m pytest iarray/tests
#  conda deactivate
#done

/opt/python/cp37-cp37m/bin/python -m pip install twine
/opt/python/cp37-cp37m/bin/python -m twine upload dist/*
