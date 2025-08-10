"""
This file generates the data used in `test/03_losses.cu`.

Usage:
    `python 03_losses.py <data_folder> <temporary_folder>`
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import numpy as np
import torch

import utils


def crossentropy():
    """
    Generates random data for testing the CrossEntropy loss function.
    """
    path = "losses/crossentropy"

    x = utils.create_tensor(np.random.uniform(1e-3, 10, size=[100, 10]))
    utils.write_tensor_to_file(x, path + "_input")

    target = utils.create_tensor(np.random.randint(0, 10, size=len(x))).type(torch.long)
    utils.write_tensor_to_file(target, path + "_target")

    y = torch.nn.functional.nll_loss(torch.log(x), target, reduction="none")
    utils.write_tensor_to_file(y, path + "_output")

    y.mean().backward()
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")


def crossentropy_special_cases():
    """
    Generates special cases for testing the CrossEntropy loss function.
    """
    path = "losses/crossentropy_special_cases"

    x = utils.create_tensor([[0.0, 0.0], [0.0, 0.0]])
    utils.write_tensor_to_file(x, path + "_input")

    target = utils.create_tensor([1, 0])
    utils.write_tensor_to_file(target, path + "_target")


def crossentropy_3D():
    """
    Generates random 3D data for testing the CrossEntropy loss function.
    """
    path = "losses/crossentropy_3D"

    batchsize = 20
    sequence_length = 12
    num_classes = 10

    x = utils.create_tensor(
        np.random.uniform(0.1, 10, size=[batchsize, sequence_length, num_classes])
    )
    utils.write_tensor_to_file(x, path + "_input")

    target = utils.create_tensor(
        np.random.randint(0, num_classes, size=[batchsize, sequence_length])
        * np.random.choice([0, 1], size=[batchsize, sequence_length], p=[0.2, 0.8])
    ).type(torch.long)
    utils.write_tensor_to_file(target, path + "_target")

    y = torch.nn.functional.nll_loss(
        torch.log(x.reshape([batchsize * sequence_length, num_classes])),
        target.reshape([batchsize * sequence_length]),
        reduction="none",
        ignore_index=0,
    )
    utils.write_tensor_to_file(
        y.reshape([batchsize, sequence_length]), path + "_output"
    )

    y.mean().backward()
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")


if __name__ == "__main__":
    crossentropy()
    crossentropy_special_cases()
    crossentropy_3D()
