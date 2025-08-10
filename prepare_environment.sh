#!/bin/bash

# This script prepares the environment for the project by loading the necessary
# software modules, such as compilers, libraries, and other tools.
#
# If the first argument is "proxy", the script also exports proxy settings to
# allow network access via a specified proxy server.
#
# Usage:
#   `. prepare_environment.sh [proxy]`

module load nvhpc

# Check available modules and load the appropriate PyTorch module
if module is-avail python/pytorch-1.13py3.10; then # for use on TinyGPU
    module load cuda/11.6.1 python/pytorch-1.13py3.10
elif module is-avail python/pytorch-1.10py3.9; then # for use on Alex
    module load cudnn/8.2.4.15-11.4 cuda/11.5.0 python/pytorch-1.10py3.9
else
    echo "No suitable PyTorch module found."
    exit 1
fi

export NVHPC_CUDA_HOME=$CUDA_HOME

# Export the proxy codes if the first argument is "proxy"
if [ "$1" = "proxy" ]; then
    export http_proxy=http://proxy:80
    export https_proxy=http://proxy:80
fi
