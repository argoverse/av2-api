#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENVIRONMENT_FILE=environment.yml

# Ensure mamba is installed.
conda install -y mamba -c conda-forge

# Create library environment.
mamba env create -f ${SCRIPT_DIR}/${ENVIRONMENT_FILE} \
&& eval "$(conda shell.bash hook)" \
&& conda activate av2 \
&& OPENSSL_DIR=$CONDA_PREFIX python -m pip install -e $SCRIPT_DIR/..
