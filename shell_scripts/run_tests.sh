#!/bin/bash

# This script runs the compiled test executables listed in
# `$TMPDIR/compiled_executables.txt`.
#    - If `MEMCHECK` is set to 1, it uses compute-sanitizer to check for memory
#      leaks.
#    - If `MEMCHECK` is set to 0, it runs the executables directly without
#      compute-sanitizer.
#    - If `CONTINUE_ON_FAIL` is set to 1, it will continue running even if a
#      test fails.
#
# Usage:
#    `./run_tests.sh <BIN_DIR> <DATA_DIR> <MEMCHECK> <CONTINUE_ON_FAIL>`
#    - `BIN_DIR`: Directory containing the compiled executables.
#    - `DATA_DIR`: Directory containing the data to be used by the executables.
#    - `MEMCHECK`: Set to 1 to use compute-sanitizer, 0 to run without it.
#    - `CONTINUE_ON_FAIL`: Set to 1 to continue running tests even if one fails,
#      0 to abort on failure.

BIN_DIR=$1
DATA_DIR=$2
MEMCHECK=$3
CONTINUE_ON_FAIL=$4

TOTAL_FAILS=0

EXECUTABLES=$(cat $TMPDIR/compiled_executables.txt)

for exec in $EXECUTABLES; do
    echo "Running $BIN_DIR/$exec"
    echo ""
    if [ $MEMCHECK -eq 1 ]; then
        compute-sanitizer --tool memcheck --leak-check full $BIN_DIR/$exec \
            $DATA_DIR $TMPDIR
    else
        $BIN_DIR/$exec $DATA_DIR $TMPDIR
    fi
    RETURN_VALUE=$?
    if [ $RETURN_VALUE -gt 0 ]; then
        TOTAL_FAILS=$((TOTAL_FAILS + RETURN_VALUE))
        if [ $CONTINUE_ON_FAIL -eq 0 ]; then
            echo ""
            echo -e "\033[31mTest failed with return value $RETURN_VALUE. Aborting...\033[0m"
            break
        fi
    fi
    echo ""
done

if [ $TOTAL_FAILS -eq 0 ]; then
    echo -e "\033[34m=== All tests passed! ===\033[0m"
else
    echo -e "\033[34m=== In total $TOTAL_FAILS test(s) failed! ===\033[0m"
    echo -e "\033[90m(This value is only accurate if the program completed without critical errors!)\033[0m"
fi
