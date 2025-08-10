#!/bin/bash

# This script compiles all `*.cpp` and `*.cu` files in the test directory. Files
# starting with `_` are excluded. The compiled executables are placed in the
# `BIN_DIR`. A list of compiled executables is saved in
# `$TMPDIR/compiled_executables.txt`.
#
# Usage:
#    `./compile_tests.sh <BIN_DIR> <CUDA_FLAGS> <TEST_DIR>`
#    - `BIN_DIR`: Directory where compiled executables will be placed.
#    - `CUDA_FLAGS`: Flags to be passed to the CUDA compiler.
#    - `TEST_DIR`: Directory containing the `*.cpp` and `*.cu` files to be
#      compiled.

BIN_DIR=$1
CUDA_FLAGS=$2
TEST_DIR=$3

mkdir -p $BIN_DIR

> $TMPDIR/compiled_executables.txt

SOURCES=$(find $TEST_DIR -name "*.cpp" -o -name "*.cu" | grep -v "/_" | sort)

for src in $SOURCES; do
    echo "Compiling $src"
    executable=$(basename $src .cpp)
    nvc++ $CUDA_FLAGS $src -o $BIN_DIR/$executable || exit 1
    echo $executable >> $TMPDIR/compiled_executables.txt
done
