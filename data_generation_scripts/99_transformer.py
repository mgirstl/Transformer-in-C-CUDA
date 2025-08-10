"""
This file generates the data used in the transformer targets of the makefile.

Usage:
    `python 99_transformer.py <data_folder> <temporary_folder>`
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import os

import numpy as np

import utils

# Transformer
train_size = 100
test_size = 100
num_embeddings = 100
num_embeddings_benchmark = 6880
sequence_length = 13

transformer_data = np.zeros([train_size, sequence_length], dtype=int)
transformer_data += np.arange(0, train_size)[:, None]
transformer_data += 3 * np.arange(0, sequence_length)[None, :]
transformer_data %= num_embeddings
transformer_target = (transformer_data + 2) % num_embeddings

indices = np.random.permutation(train_size)
transformer_data = transformer_data[indices]
transformer_target = transformer_target[indices]

utils.write_tensor_to_file(transformer_data, "transformer/train_data")
utils.write_tensor_to_file(transformer_target, "transformer/train_target")

indices = np.random.permutation(train_size)[:test_size]
utils.write_tensor_to_file(transformer_data[indices], "transformer/test_data")
utils.write_tensor_to_file(transformer_target[indices], "transformer/test_target")

os.makedirs(f"{utils.data_folder}/transformer", exist_ok=True)

with open(f"{utils.data_folder}/transformer/config", "w") as file:
    file.write(f"train_data_path = {utils.data_folder}/transformer/train_data\n")
    file.write(f"train_target_path = {utils.data_folder}/transformer/train_target\n")
    file.write(f"test_data_path = {utils.data_folder}/transformer/test_data\n")
    file.write(f"test_target_path = {utils.data_folder}/transformer/test_target\n")
    file.write("batchsize = 50\n")
    file.write("iterations = 200\n")
    file.write("learning_rate = 1e-3\n")
    file.write(f"sequence_length = {sequence_length}\n")
    file.write("embedding_dim = 512\n")
    file.write("num_encoder_layers = 6\n")
    file.write("num_decoder_layers = 6\n")
    file.write("num_heads = 8\n")
    file.write(f"num_embeddings = {num_embeddings}\n")
    file.write("hidden_dim = 2048\n")
    file.write("max_batchsize = 100\n")
    file.write("ignore_index = 0\n")
    file.write("dropout = 0.1\n")
    file.write("warmup_steps = 50\n")
    file.write("tf32 = 1\n")

with open(f"{utils.data_folder}/transformer/config_benchmark", "w") as file:
    file.write(f"train_data_path = {utils.data_folder}/transformer/train_data\n")
    file.write(f"train_target_path = {utils.data_folder}/transformer/train_target\n")
    file.write(f"test_data_path = {utils.data_folder}/transformer/test_data\n")
    file.write(f"test_target_path = {utils.data_folder}/transformer/test_target\n")
    file.write("batchsize = 100\n")
    file.write("iterations = 200\n")
    file.write("learning_rate = 1e-3\n")
    file.write(f"sequence_length = {sequence_length}\n")
    file.write("embedding_dim = 512\n")
    file.write("num_encoder_layers = 6\n")
    file.write("num_decoder_layers = 6\n")
    file.write("num_heads = 8\n")
    file.write(f"num_embeddings = {num_embeddings_benchmark}\n")
    file.write("hidden_dim = 2048\n")
    file.write("max_batchsize = 100\n")
    file.write("ignore_index = 0\n")
    file.write("dropout = 0.1\n")
    file.write("benchmark = 1\n")
    file.write("warmup_steps = 50\n")
    file.write("tf32 = 1\n")

with open(f"{utils.data_folder}/transformer/config_test", "w") as file:
    file.write(f"train_data_path = {utils.data_folder}/transformer/train_data\n")
    file.write(f"train_target_path = {utils.data_folder}/transformer/train_target\n")
    file.write(f"test_data_path = {utils.data_folder}/transformer/test_data\n")
    file.write(f"test_target_path = {utils.data_folder}/transformer/test_target\n")
    file.write("batchsize = 100\n")
    file.write("iterations = 50\n")
    file.write("learning_rate = 1e-3\n")
    file.write(f"sequence_length = {sequence_length}\n")
    file.write("embedding_dim = 512\n")
    file.write("num_encoder_layers = 6\n")
    file.write("num_decoder_layers = 6\n")
    file.write("num_heads = 8\n")
    file.write(f"num_embeddings = {num_embeddings_benchmark}\n")
    file.write("hidden_dim = 2048\n")
    file.write("max_batchsize = 100\n")
    file.write("ignore_index = 0\n")
    file.write("dropout = 0.1\n")
    file.write("benchmark = 1\n")
    file.write("warmup_steps = 0\n")
    file.write("tf32 = 1\n")
