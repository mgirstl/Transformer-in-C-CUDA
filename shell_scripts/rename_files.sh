#!/bin/bash

# This script renames files in the specified directory. If `ACTION` is
# "exclude", it adds an underscore (_) to the beginning of filenames that start
# with digits. If `ACTION` is "include", it removes the leading underscore (_)
# from filenames that start with an underscore followed by digits.
#
# Usage:
#    `./rename_files.sh <DIRECTORY> <ACTION>`
#    - `DIRECTORY`: The directory containing the files to be renamed.
#    - `ACTION`: The action to perform ("exclude" to add an underscore,
#      "include" to remove an underscore)

DIRECTORY=$1
ACTION=$2

for file in "$DIRECTORY"/*; do
    filename=$(basename "$file")
    if [[ "$ACTION" == "exclude" && "$filename" =~ ^[0-9] ]]; then
        mv "$file" "$DIRECTORY/_$filename"
    elif [[ "$ACTION" == "include" && "$filename" =~ ^_[0-9] ]]; then
        mv "$file" "$DIRECTORY/${filename:1}"
    fi
done
