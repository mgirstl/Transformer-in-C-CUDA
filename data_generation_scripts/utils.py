"""
This file implements helper functions to create and manage data files for
testing and general use with the C++ codebase implemented in this repository for
building neural networks (e.g., solving MNIST or implementing transformers).

Usage:
    python utils.py <data_folder> <temporary_folder>
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import os
import random
import sys

import numpy as np
import torch

# Read Command Line Arguments
if len(sys.argv) < 3:
    raise RuntimeError(
        "Please provide the path to the data directory as first command-line argument and the path to a temporary directory as second command-line argument."
    )

data_folder = sys.argv[1]
temporary_folder = sys.argv[2]

# Set Random Seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def write_tensor_to_file(tensor, filename):
    """
    Writes a tensor to a file in a custom format.

    Parameters:
        tensor: The tensor to write to the file. It can be any input that can be
        cast to a numpy array, including scalars, lists, and torch tensors.
        filename: The name of the file to write the tensor to.
    """
    path = data_folder + "/" + filename + ".tensor"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as file:
        if not isinstance(tensor, torch.Tensor):
            tensor = np.asarray(tensor, dtype=float)

        # rank
        file.write(str(tensor.ndim) + "\n")

        # shape
        if tensor.shape:
            shape = " ".join(map(str, tensor.shape))
            file.write(shape + "\n")

        # elements
        if tensor.size:
            elements = " ".join(map(str, tensor.flatten().tolist()))
            file.write(elements)


def create_tensor(array):
    """
    Creates a tensor from an input that can be cast to a numpy array and sets
    requires_grad to True.

    Parameters:
        array: The input to convert to a tensor. It can be any input that can be
         cast to a numpy array, including scalars and lists.

    Returns:
        torch.Tensor: The created tensor with requires_grad set to True.
    """
    tensor = torch.from_numpy(np.asarray(array, dtype=float))
    tensor.requires_grad = True
    return tensor


def print_tensor(tensor, name="", max_show=7):
    """
    Prints a string representation of a NumPy array or PyTorch tensor with a
    specified name.

    Parameters:
    tensor (np.ndarray or torch.Tensor): The tensor to be printed.
    name (str): The name to be printed along with the tensor representation.
    max_show (int): The maximum number of elements to show in the data
    representation.

    Raises:
    TypeError: If the input is not a NumPy array or a PyTorch tensor.
    """
    if isinstance(tensor, np.ndarray):
        data = tensor

    elif isinstance(tensor, torch.Tensor):
        data = tensor.detach().numpy()

    elif tensor is None:
        print(name, "None")
        return

    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor!")

    rank = data.ndim
    shape = data.shape
    size = data.size

    string = "{"
    string += f"rank: {rank}, "

    if rank:
        string += "shape: [" + ", ".join(map(str, shape)) + "], "
    else:
        string += "[], "

    string += f"size: {size}, "

    if size:
        flat_data = [f"{x:.5g}" for x in data.flat]
        string += "data: ["
        if size < max_show:
            string += ", ".join(flat_data)
        else:
            string += ", ".join(flat_data[: max_show // 2])
            string += ", ..., "
            string += ", ".join(flat_data[-max_show // 2 :])
        string += "]}"
    else:
        string += "[]}"

    if name != "" and name[-1] != ":":
        name += ":"

    print(name, string)
