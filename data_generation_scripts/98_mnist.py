"""
This file generates the data used in the mnist targets of the makefile.

Usage:
    `python 98_mnist.py <data_folder> <temporary_folder>`
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import os
import sys
import warnings

import torchvision

import utils

os.makedirs(f"{utils.data_folder}/mnist", exist_ok=True)

with open(f"{utils.data_folder}/mnist/config", "w") as file:
    file.write(f"train_data_path = {utils.data_folder}/mnist/train_data\n")
    file.write(f"train_target_path = {utils.data_folder}/mnist/train_target\n")
    file.write(f"test_data_path = {utils.data_folder}/mnist/test_data\n")
    file.write(f"test_target_path = {utils.data_folder}/mnist/test_target\n")
    file.write("input_1D = 100\n")
    file.write("hidden_dim = 1000\n")
    file.write("batchsize = 100\n")
    file.write("epochs = 5\n")
    file.write("learning_rate = 1e-3\n")
    file.write("warmup_steps = 50")

with open(f"{utils.data_folder}/mnist/config_benchmark", "w") as file:
    file.write(f"train_data_path = {utils.data_folder}/mnist/train_data\n")
    file.write(f"train_target_path = {utils.data_folder}/mnist/train_target\n")
    file.write(f"test_data_path = {utils.data_folder}/mnist/test_data_small\n")
    file.write(f"test_target_path = {utils.data_folder}/mnist/test_target_small\n")
    file.write("input_1D = 400\n")
    file.write("hidden_dim = 15000\n")
    file.write("batchsize = 100\n")
    file.write("iterations = 200\n")
    file.write("learning_rate = 1e-3\n")
    file.write("benchmark = 1\n")
    file.write("warmup_steps = 50")

all_exist = True
for filename in [
    "train_data",
    "train_target",
    "test_data",
    "test_target",
    "test_data_small",
    "test_target_small",
]:
    if not os.path.exists(f"{utils.data_folder}/mnist/{filename}.tensor"):
        all_exist = False

if all_exist:
    sys.exit(0)

warnings.warn(
    "A working internet connection is required! If the script takes too long, please check your network settings.",
    UserWarning
)

train_dataset = torchvision.datasets.MNIST(
    root=f"{utils.temporary_folder}/mnist", train=True, download=True
)
utils.write_tensor_to_file(train_dataset.data / 255.0, "mnist/train_data")
utils.write_tensor_to_file(train_dataset.targets, "mnist/train_target")

test_dataset = torchvision.datasets.MNIST(
    root=f"{utils.temporary_folder}/mnist", train=False, download=True
)
utils.write_tensor_to_file(test_dataset.data / 255.0, "mnist/test_data")
utils.write_tensor_to_file(test_dataset.targets, "mnist/test_target")
utils.write_tensor_to_file(test_dataset.data[:1000] / 255.0, "mnist/test_data_small")
utils.write_tensor_to_file(test_dataset.targets[:1000], "mnist/test_target_small")
