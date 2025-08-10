#!/bin/bash

# This script is used to allocate GPU resources on different types of machines.
# It checks the hostname and based on the prefix of the hostname, it allocates
# the appropriate GPU resources for the specified time.
#
# Arguments:
#   - gpu_type (optional): Specifies the type of GPU to allocate. Options are
#     a40, a100, work, rtx3080, v100, or default. If no argument is provided,
#     the default option is used.
#   - gpu_model (optional): Specifies the model of the GPU. This is only
#     applicable if gpu_type is a100. In this case, you can specify 80gb or 80GB
#     to allocate an A100 GPU with 80GB memory. If not specified, it defaults to
#     using the 40GB version.
#   - time_in_minutes (optional): Specifies the time in minutes for the
#     allocation. If not specified, the default is 30 minutes.
#
# Notes:
#   - If the allocation takes a long time, please check if nodes are available
#     with the `sinfo` command.
#
# Usage:
#   `. alloc.sh [gpu_type] [gpu_model] [time_in_minutes]`

HOSTNAME=$(hostname)
PREFIX=${HOSTNAME%%[0-9]*}
TIME_IN_MINUTES=30

# Check if the third argument is a number (time in minutes)
if [[ $3 =~ ^[0-9]+$ ]]; then
    TIME_IN_MINUTES=$3
elif [[ $2 =~ ^[0-9]+$ ]]; then
    TIME_IN_MINUTES=$2
elif [[ $1 =~ ^[0-9]+$ ]]; then
    TIME_IN_MINUTES=$1
fi

# Convert time in minutes to HH:MM:SS format
HOURS=$((TIME_IN_MINUTES / 60))
MINUTES=$((TIME_IN_MINUTES % 60))
ARGS="--time=$(printf "%02d:%02d:00" $HOURS $MINUTES)"

# Select and allocate a node
if [ "$PREFIX" == "alex" ]; then
    if [ "$1" == "a40" ]; then
        salloc --gres=gpu:a40:1 $ARGS
    elif ([ "$1" == "a100" ] && \
          ([ "$2" == "80gb" ] || [ "$2" == "80GB" ])); then
        salloc --gres=gpu:a100:1 -C a100_80 $ARGS
    elif [ "$1" == "a100" ] || [ "$1" == "default" ]; then
        salloc --gres=gpu:a100:1 $ARGS
    else
        salloc --gres=gpu:a100:1 $ARGS
    fi
fi

if [ "$PREFIX" == "tinyx" ]; then
    if [ "$1" == "work" ] || [ "$1" == "default" ]; then
        salloc.tinygpu --gres=gpu:1 $ARGS
    elif [ "$1" == "rtx3080" ]; then
        salloc.tinygpu --gres=gpu:1 -p rtx3080 $ARGS
    elif [ "$1" == "a100" ]; then
        salloc.tinygpu --gres=gpu:a100:1 -p a100 $ARGS
    elif [ "$1" == "v100" ]; then
        salloc.tinygpu --gres=gpu:v100:1 -p v100 $ARGS
    else
        salloc.tinygpu --gres=gpu:1 $ARGS
    fi
fi
