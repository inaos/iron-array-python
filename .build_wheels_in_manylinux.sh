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
versions=(cp37-cp37m)

for version in "${versions[@]}"; do
  rm -rf _skbuild/ iarray/iarray-c-develop/build/*
  /opt/python/${version}/bin/python -m pip install --upgrade pip
  #/opt/python/${version}/bin/python -m pip install cython ninja cmake scikit-build auditwheel
  /opt/python/${version}/bin/python -m pip install cython numpy 
  /opt/python/${version}/bin/python setup.py build -j 4 --build-type RelWithDebInfo  -- -DDISABLE_LLVM_CONFIG=True -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
  # Copy the necessary shared libraries 
  /bin/cp -f iarray/iarray-c-develop/build/libiarray.so iarray/
  pushd iarray
  patchelf --remove-needed libintlc.so.5 libiarray.so
  patchelf --remove-needed libsvml.so libsvml.so
  patchelf --add-needed ./libintlc.so.5 libiarray.so
  patchelf --add-needed ./libsvml.so libsvml.so
  popd
  /opt/python/${version}/bin/python setup.py bdist_wheel
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:iarray
for whl in dist/*linux_*.whl; do
  /opt/python/cp37-cp37m/bin/auditwheel repair ${whl} -w /work/dist/  --plat manylinux2014_x86_64
  rm ${whl}
done

for version in "${versions[@]}"; do
  pybin=/opt/python/${version}/bin/python
  ${pybin} -m pip install --user pytest llvmlite numexpr
  cd /tmp/
  ${pybin} -m pip install iron-array-python --user --no-cache-dir --no-index -f /work/dist/
  cd /work/
  ${pybin} iron-array-python/test.py
done
