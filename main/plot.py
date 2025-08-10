"""
This script plots the benchmarking results of the transformer_scaling target in
the Makefile.

It reads log files from a specified directory, extracts relevant data, and
generates plots for various parameters.

Usage:
    `python script.py <log_folder>`
    - `log_directory`: The path to the directory where the log files are stored.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Check if the path to the log directory is provided as a command-line argument
if len(sys.argv) < 2:
    raise RuntimeError(
        "Please provide the path to the log directory as first command-line argument!"
    )

# Define the paths for the log files and the folder to save the plots
log_folder = f"{sys.argv[1]}/transformer_scaling"
save_folder = f"{sys.argv[1]}/plots"
os.makedirs(save_folder, exist_ok=True)

# Set the default parameters for the plots
rcParams["font.size"] = 16
rcParams["figure.figsize"] = [16 / 2.54, 13 / 2.54]
rcParams["savefig.format"] = "png"
rcParams["figure.dpi"] = 300
rcParams["axes.facecolor"] = "white"
rcParams["savefig.facecolor"] = (0.0, 0.0, 0.0, 0.0)
rcParams["figure.facecolor"] = (0.0, 0.0, 0.0, 0.0)


def load(parameter):
    """
    Load data from log files.

    Args:
        parameter (str): The parameter to load data for.

    Returns:
        tuple: Arrays of hyper parameters, runtime, number of parameters,
        calculated memory, and measured memory.
    """
    path = f"{log_folder}/{parameter}.log"

    hyper_parameter = []
    runtime = []
    num_parameters = []
    calculated_mem = []
    measured_mem = []

    with open(path, "r") as file:
        for line in file:
            tokens = line.split()
            key = tokens[0]

            if key == f"{parameter}:":
                hyper_parameter.append(int(tokens[-1]))
            elif key == "Runtime":
                runtime.append(float(tokens[-1]))
            elif key == "Parameters:":
                num_parameters.append(int(tokens[-1]))
            elif key == "Calculated":
                calculated_mem.append(float(tokens[-1]))
            elif key == "Measured":
                measured_mem.append(float(tokens[-1]))

    return (
        np.asarray(hyper_parameter),
        np.asarray(runtime),
        np.asarray(num_parameters),
        np.asarray(calculated_mem),
        np.asarray(measured_mem),
    )


def plot(path, y_name, x1_name, x1, y1, x2_name, x2, y2, sync_ticks=False):
    """
    Plot the data.

    Args:
        path (str): The path to save the plot.
        y_name (str): The name of the y-axis.
        x1_name (str): The name of the first x-axis.
        x1 (array): The data for the first x-axis.
        y1 (array): The data for the y-axis corresponding to the first x-axis.
        x2_name (str): The name of the second x-axis.
        x2 (array): The data for the second x-axis.
        y2 (array): The data for the y-axis corresponding to the second x-axis.
        sync_ticks (bool): Whether to synchronize the ticks of the two x-axes.
    """
    fig, ax = plt.subplots()

    color = "#00b0f0"
    ax.plot(x1, y1, color=color, ls="--", marker="o")
    ax.set_xlabel(x1_name, color=color)
    ax.set_ylabel(y_name)
    ax.spines["bottom"].set_color(color)
    ax.tick_params(axis="x", colors=color)
    ax.grid()

    color = "#e97132"
    ax2 = ax.twiny()
    ax2.plot(x2, y2, color=color, ls="--", marker="o")
    ax2.set_xlabel(x2_name, color=color)
    ax2.spines["top"].set_color(color)
    ax2.tick_params(axis="x", colors=color)

    if sync_ticks:
        ax2.set_xlim(ax.get_xlim())

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{path}")


def plot_all(path, name_1, parameter_1, name_2, parameter_2, **kwargs):
    """
    Plot all the data for the given parameters.

    Args:
        path (str): The base path to save the plots.
        name_1 (str): The name of the first parameter.
        parameter_1 (str): The first parameter.
        name_2 (str): The name of the second parameter.
        parameter_2 (str): The second parameter.
        **kwargs: Additional arguments for the plot function.
    """
    x1, runtime_1, parameters_1, calculated_mem_1, measured_mem_1 = load(parameter_1)
    x2, runtime_2, parameters_2, calculated_mem_2, measured_mem_2 = load(parameter_2)
    plot(
        path + "_runtime",
        "Runtime [ms]",
        name_1,
        x1,
        runtime_1,
        name_2,
        x2,
        runtime_2,
        **kwargs,
    )
    plot(
        path + "_parameters",
        "# Trainable Parameters",
        name_1,
        x1,
        parameters_1,
        name_2,
        x2,
        parameters_2,
        **kwargs,
    )
    plot(
        path + "_calculated_mem",
        "Theoretical Memory Usage [MB]",
        name_1,
        x1,
        calculated_mem_1,
        name_2,
        x2,
        calculated_mem_2,
        **kwargs,
    )
    plot(
        path + "_measured_mem",
        "Measured Memory Usage [MB]",
        name_1,
        x1,
        measured_mem_1,
        name_2,
        x2,
        measured_mem_2,
        **kwargs,
    )


# Batchsize and Sequence_Length
plot_all(
    "batchsize",
    "Batch Size   (Sequence Length = 13)",
    "batchsize",
    "Sequence Length   (Batch Size = 100)",
    "sequence_length"
)

# Num_Encoder_Layers and Num_Decoder_Layers
plot_all(
    "encoder",
    "# Encoder Layers   (# Decoder Layers = 6)",
    "num_encoder_layers",
    "# Decoder Layers   (# Encoder Layers = 6)",
    "num_decoder_layers",
    sync_ticks=True,
)

# Embedding_Dim and Hidden_Dim
plot_all(
    "hidden",
    "Hidden Dimension Length   (EDL = 512)",
    "hidden_dim",
    "Embedding Dimension Length   (HDL = 2048)",
    "embedding_dim",
)

# Num_Heads and Num_Embeddings
plot_all(
    "heads",
    "# Attention Heads   (# Embeddings = 6880)",
    "num_heads",
    "# Embeddings   (# Attention Heads = 8)",
    "num_embeddings"
)
