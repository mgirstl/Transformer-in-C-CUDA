"""
This file generates the data used in `test/07_models.cu`.

Usage:
    `python 07_models.py <data_folder> <temporary_folder>`
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import os

import numpy as np

import utils


def mnist():
    """
    Generates random data for testing the MNIST and MNIST_Extended model.
    """
    path = "models/mnist"

    data = np.random.uniform(0, 1, size=[50, 28, 28])
    utils.write_tensor_to_file(data, path + "_data")

    target = np.random.randint(0, 10, size=[50])
    utils.write_tensor_to_file(target, path + "_target")

    data_large = np.repeat(data, 3, axis=0)
    utils.write_tensor_to_file(data_large, path + "_data_large")

    target_large = np.repeat(target, 3, axis=0)
    utils.write_tensor_to_file(target_large, path + "_target_large")

    path = f"{utils.data_folder}/models"
    os.makedirs(path, exist_ok=True)

    path += "/mnist"

    with open(path + "_config", "w") as file:
        file.write(f"train_data_path = {path}_data\n")
        file.write(f"train_target_path = {path}_target\n")
        file.write(f"test_data_path = {path}_data\n")
        file.write(f"test_target_path = {path}_target\n")
        file.write("input_1D = 100\n")
        file.write("hidden_dim = 1000\n")
        file.write("batchsize = 100\n")
        file.write("iterations = 500\n")
        file.write("learning_rate = 1e-3")


def transformer():
    """
    Generates data for testing the Transformer model.
    """
    path = "models/transformer"

    batchsize = 50
    num_embeddings = 50
    sequence_length = 10

    data = np.zeros([batchsize, sequence_length], dtype=int)
    data += np.arange(0, batchsize)[:, None]
    data += 3 * np.arange(0, sequence_length)[None, :]
    data %= num_embeddings
    utils.write_tensor_to_file(data, path + "_data")

    target = (data + 2) % num_embeddings
    utils.write_tensor_to_file(target, path + "_target")

    path = f"{utils.data_folder}/models"
    os.makedirs(path, exist_ok=True)

    path += "/transformer"

    with open(path + "_config", "w") as file:
        file.write(f"train_data_path = {path}_data\n")
        file.write(f"train_target_path = {path}_target\n")
        file.write(f"test_data_path = {path}_data\n")
        file.write(f"test_target_path = {path}_target\n")
        file.write("batchsize = 100\n")
        file.write("iterations = 1000\n")
        file.write("learning_rate = 1e-3\n")
        file.write(f"sequence_length = {sequence_length}\n")
        file.write("embedding_dim = 32\n")
        file.write("num_encoder_layers = 2\n")
        file.write("num_decoder_layers = 2\n")
        file.write("num_heads = 8\n")
        file.write(f"num_embeddings = {num_embeddings}\n")
        file.write("hidden_dim = 32\n")
        file.write("max_batchsize = 100\n")
        file.write("ignore_index = 0\n")


if __name__ == "__main__":
    mnist()
    transformer()
