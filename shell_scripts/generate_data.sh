#!/bin/bash

# This script runs all the Python scripts in `DATA_GENERATION_SCRIPTS_DIR`.
# Files starting with `_` are excluded. It executes each script found and passes
# the `DATA_DIR` as a command line argument. If any script fails, the operation
# will abort.
#
# Usage:
#    `./run_data_generation.sh <DATA_DIR> <DATA_GENERATION_SCRIPTS_DIR>`
#    - `DATA_DIR`: Directory where data will be stored.
#    - `DATA_GENERATION_SCRIPTS_DIR`: Directory containing the Python scripts to
#      be executed.

DATA_DIR=$1
DATA_GENERATION_SCRIPTS_DIR=$2

mkdir -p $DATA_DIR

SCRIPTS=$(ls -v $DATA_GENERATION_SCRIPTS_DIR/*.py | grep -v "/_")

for script in $SCRIPTS; do
    echo "Running $script"
    python $script $DATA_DIR $TMPDIR || exit 1
done
