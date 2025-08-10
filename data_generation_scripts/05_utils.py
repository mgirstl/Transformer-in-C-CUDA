"""
This file generates the data used in `test/05_utils.cu`.

Usage:
    `python 05_utils.py <data_folder> <temporary_folder>`
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import os

import numpy as np

import utils


def dataset_handler():
    """
    Generates data for testing the DatasetHandler functionality.
    """
    path = "utils"

    batchsize = 3
    utils.write_tensor_to_file(batchsize, f"{path}/batchsize")

    data = [
        [[1, 1], [1, 1]],
        [[2, 2], [2, 2]],
        [[3, 3], [3, 3]],
        [[4, 4], [4, 4]],
        [[5, 5], [5, 5]],
    ]

    utils.write_tensor_to_file(data, f"{path}/data")
    utils.write_tensor_to_file(data[:batchsize], f"{path}/data_first_batch")
    utils.write_tensor_to_file(
        data[batchsize:] + data[: 2 * batchsize - len(data)],
        f"{path}/data_second_batch",
    )

    target = [1, 2, 3, 4, 5]
    utils.write_tensor_to_file(target, f"{path}/target")
    utils.write_tensor_to_file(target[:batchsize], f"{path}/target_first_batch")
    utils.write_tensor_to_file(
        target[batchsize:] + target[: 2 * batchsize - len(target)],
        f"{path}/target_second_batch",
    )


def config():
    """
    Generates configuration data for testing.
    """
    path = f"{utils.data_folder}/utils"

    os.makedirs(path, exist_ok=True)

    with open(f"{path}/config", "w") as file:
        file.write("# this is a comment\n")
        file.write("train_data_path = path/to/train_data\n")
        file.write("train_target_path=path/to/train_target\n")
        file.write("test_data_path =path/to/test_data\n")
        file.write("test_target_path= path/to/test_target\n")
        file.write("output_path = path to output\n")
        file.write("\n")
        file.write("  batchsize = 10\n")
        file.write("epochs = 5  \n")
        file.write("iterations = 10000\n")
        file.write("learning_rate = 1e-3 # some inline comment")

    with open(f"{path}/config_expected_values", "w") as file:
        file.write("path/to/train_data\n")
        file.write("path/to/train_target\n")
        file.write("path/to/test_data\n")
        file.write("path/to/test_target\n")
        file.write("path\n")
        file.write("10\n")
        file.write("5\n")
        file.write("10000\n")
        file.write("1e-3")


def indicator():
    """
    Generates random data for testing the indicator functionality.
    """
    path = "utils/indicator"

    x = np.random.randint(0, 5, size=[100, 13])
    utils.write_tensor_to_file(x, path + "_input")

    target = np.random.randint(0, 5, size=x.shape)
    utils.write_tensor_to_file(target, path + "_target")

    y = (x == target).astype(int)
    utils.write_tensor_to_file(y, path + "_output")


def argmax():
    """
    Generates random data for testing the argmax functionality.
    """
    path = "utils/argmax"

    x = np.random.uniform(-10, 10, size=[50, 10, 13])
    utils.write_tensor_to_file(x, path + "_input")

    y = np.argmax(x, axis=-1)
    utils.write_tensor_to_file(y, path + "_output")


if __name__ == "__main__":
    dataset_handler()
    config()
    indicator()
    argmax()
