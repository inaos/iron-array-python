#!/bin/bash

set -x

mkdir -p $HOME/.inaos/cmake
mkdir -p $HOME/INAOS
echo "INAC_REPOSITORY_LOCAL=$HOME/INAOS" > $HOME/.inaos/cmake/repository.txt
echo "INAC_REPOSITORY_REMOTE=https://inaos.jfrog.io/inaos/libs-release-local/inaos" >> $HOME/.inaos/cmake/repository.txt
echo "INAC_REPOSITORY_USRPWD=licensed:AKCp5bBraH7CasbsYCURsjzkbjXwVwdYcT7u39EiuL6GjnK1VKfKQWCd1E2E64mHokU5YUHku" >> $HOME/.inaos/cmake/repository.txt

# Activate the conda environment in this docker image
source /work/conda/etc/profile.d/conda.sh
conda activate

########### Original code starts here ###########################
#versions=(cp36-cp36m cp37-cp37m cp38-cp38)
#versions=(cp37-cp37m)
versions=(cp38-cp38)

for version in "${versions[@]}"; do
  /opt/python/${version}/bin/python -m pip install --upgrade pip
  /opt/python/${version}/bin/python -m pip install cython numpy 
  rm -rf _skbuild/ iarray/iarray-c-develop/build/*  # uncomment in production...
  /opt/python/${version}/bin/python setup.py build -j 4 --build-type RelWithDebInfo  -- -DDISABLE_LLVM_CONFIG=True -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
  # Copy the necessary shared libraries 
  /bin/cp -f iarray/iarray-c-develop/build/libiarray.so iarray/
  /opt/python/${version}/bin/python setup.py bdist_wheel --plat-name manylinux2014_x86_64
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:iarray
for whl in dist/*linux*.whl; do
  /opt/python/cp37-cp37m/bin/auditwheel show ${whl}
  # Looks like reparing the wheel only throws more unnecessary libs into the wheel 
  # /opt/python/cp37-cp37m/bin/auditwheel repair ${whl} -w /work/dist/  --plat manylinux2014_x86_64
done

for version in "${versions[@]}"; do
  pybin=/opt/python/${version}/bin/python
  ${pybin} -m pip install --user pytest llvmlite numexpr
  cd /tmp/
  ${pybin} -m pip install iarray --user --no-cache-dir --no-index -f /work/dist/
  cd /work/
  ${pybin} -m pytest iarray/tests
done
