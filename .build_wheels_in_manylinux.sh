#!/bin/bash

set -x

export additional_channel="--add channels defaults"

if [ "$(uname -m)" = "x86_64" ]; then
   export supkg="su-exec"
   #export condapkg="https://github.com/conda-forge/miniforge/releases/download/4.8.3-0/Miniforge3-4.8.3-0-Linux-x86_64.sh"
   #export conda_chksum="1db6013e836da2ea817a53c44b0fd9beea521013bcb94b2b5440b1a61ba8b338"
   export condapkg="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
   export conda_chksum="957d2f0f0701c3d1335e3b39f235d197837ad69a944fa6f5d8ad2c686b69df3b"
else
   exit 1
fi

# give sudo permission for conda user to run yum (user creation is postponed
# to the entrypoint, so we can create a user with the same id as the host)
#echo 'conda ALL=NOPASSWD: /usr/bin/yum' >> /etc/sudoers

# Install the latest Miniconda with Python 3 and update everything.
curl -s -L $condapkg > miniconda.sh
sha256sum miniconda.sh | grep $conda_chksum

rm -rf $PWD/conda
bash miniconda.sh -b -p $PWD/conda
rm -f miniconda.sh
touch $PWD/conda/conda-meta/pinned
ls -R 
source $PWD/conda/etc/profile.d/conda.sh
conda activate
#conda config --set show_channel_urls True
#conda config ${additional_channel} --add channels conda-forge
conda config --show-sources
conda update --all --yes
conda clean -tipy

# Install conda build and deployment tools.
conda install --yes --quiet llvmdev
conda clean -tipy

# # Install docker tools
# conda install --yes $supkg
# export CONDA_SUEXEC_INFO=( `conda list $supkg | grep $supkg` )
# echo "$supkg ${CONDA_SUEXEC_INFO[1]}" >> $PWD/conda/conda-meta/pinned

# conda install --yes tini
# export CONDA_TINI_INFO=( `conda list tini | grep tini` )
# echo "tini ${CONDA_TINI_INFO[1]}" >> $PWD/conda/conda-meta/pinned
# conda clean -tipy

# Lucky group gets permission to write in the conda dir
# groupadd -g 32766 lucky
# chown -R $USER $HOME/conda
# chgrp -R lucky $HOME/conda && chmod -R g=u $HOME/conda

########### Original code starts here ###########################
#versions=(cp36-cp36m cp37-cp37m cp38-cp38)
versions=(cp37-cp37m)

for version in "${versions[@]}"; do
  /opt/python/${version}/bin/python -m pip install --upgrade pip
  /opt/python/${version}/bin/python -m pip install cython ninja cmake scikit-build auditwheel
  /opt/python/${version}/bin/python setup.py -j 4 --build-type RelWithDebInfo build
  # Copy the necessary shared libraries 
  cp iarray/iarray-c-develop/build/libiarray.so iarray/
  /opt/python/${version}/bin/python setup.py bdist_wheel
done

for whl in dist/*linux_*.whl; do
  /opt/python/cp37-cp37m/bin/auditwheel repair ${whl} -w /work/dist/
  rm ${whl}
done

for version in "${versions[@]}"; do
  pybin=/opt/python/${version}/bin/python
  ${pybin} -m pip install --user numpy pytest llvmlite numexpr
  cd /tmp/
  ${pybin} -m pip install iron-array-python --user --no-cache-dir --no-index -f /work/dist/
  cd /work/
  ${pybin} iron-array-python/test.py
done
