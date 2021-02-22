#!/usr/bin/env bash

set -x

export additional_channel="--add channels defaults"

if [ "$(uname -m)" = "x86_64" ]; then
   export supkg="su-exec"
   #export condapkg="https://github.com/conda-forge/miniforge/releases/download/4.8.3-0/Miniforge3-4.8.3-0-Linux-x86_64.sh"
   #export conda_chksum="1db6013e836da2ea817a53c44b0fd9beea521013bcb94b2b5440b1a61ba8b338"
   export condapkg="https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh"
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

rm -rf /work/conda
bash miniconda.sh -b -p /work/conda
rm -f miniconda.sh
touch /work/conda/conda-meta/pinned
source /work/conda/etc/profile.d/conda.sh
conda activate
#conda config --set show_channel_urls True
#conda config ${additional_channel} --add channels conda-forge
conda config --show-sources
conda update --all --yes
conda clean -tipy

# Install conda build and deployment tools.
conda install -y --quiet llvmdev
conda install -y --quiet -c intel mkl-include
conda install -y --quiet -c intel mkl-static
conda install -y --quiet -c intel icc_rt

conda clean -tipy
