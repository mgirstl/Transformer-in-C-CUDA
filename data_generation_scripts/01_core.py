"""
This file generates the data used in `test/01_core.cu`.

Usage:
    `python 01_core.py <data_folder> <temporary_folder>`
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import numpy as np

import utils


def load():
    """
    Generates data for testing the load functionality of the tensor classes.
    """
    scalar = np.array(42)
    utils.write_tensor_to_file(scalar, "core/scalar_01")

    tensor = np.array([0, 1, 2, 3, 4, 5]).reshape([2, 3])
    utils.write_tensor_to_file(tensor, "core/tensor_01")


if __name__ == "__main__":
    load()
