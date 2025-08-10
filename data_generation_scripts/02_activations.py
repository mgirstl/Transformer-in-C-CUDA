"""
This file generates the data used in `test/02_activations.cu`.

Usage:
    `python 02_activations.py <data_folder> <temporary_folder>`
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import numpy as np
import torch

import utils


def relu():
    """
    Generates random data for testing the ReLU activation function.
    """
    path = "activations/relu"

    x = utils.create_tensor(np.random.uniform(-10, 10, size=[100, 4, 5]))
    utils.write_tensor_to_file(x, path + "_input")

    y = torch.nn.functional.relu(x)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-10, 10, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")


def softmax_2D():
    """
    Generates random 2D data for testing the softmax activation function.
    """
    path = "activations/softmax_2D"

    x = utils.create_tensor(np.random.uniform(-10, 100, size=[100, 10]))
    utils.write_tensor_to_file(x, path + "_input")

    y = torch.nn.functional.softmax(x, dim=-1)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-10, 100, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")


def softmax_10D():
    """
    Generates random 10D data for testing the softmax activation function.
    """
    path = "activations/softmax_10D"

    x = utils.create_tensor(np.random.uniform(-1, 1, size=[2] * 10))
    utils.write_tensor_to_file(x, path + "_input")

    y = torch.nn.functional.softmax(x, dim=-1)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")


def softmax_special_cases():
    """
    Generates special cases for testing the softmax activation function.
    """
    path = "activations/softmax_special_cases"

    x = utils.create_tensor(
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [1, 1, 1],
                [1e20, 1, 1],
                [1e-20, 1, 1],
                [-1e100, -1e100, -1e100],
            ]
        )
    )
    utils.write_tensor_to_file(x, path + "_input")

    y = torch.nn.functional.softmax(x, dim=-1)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-10, 10, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")


def softmax_single_value():
    """
    Generates a single value for testing the softmax activation function.
    """
    path = "activations/softmax_single_value"

    x = utils.create_tensor(np.array([10]))
    utils.write_tensor_to_file(x, path + "_input")

    y = torch.nn.functional.softmax(x, dim=-1)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.array([9]))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")


if __name__ == "__main__":
    relu()
    softmax_2D()
    softmax_10D()
    softmax_special_cases()
    softmax_single_value()
