"""
This file generates the data used in `test/04_optimizers.cu`.

Usage:
    `python 04_optimizers.py <data_folder> <temporary_folder>`
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import numpy as np
import torch

import utils


def none_optimizer():
    """
    Generates random data for testing the None optimizer.
    """
    path = "optimizers/none"

    weights = utils.create_tensor(np.random.uniform(-10, 10, size=[99, 13]))
    utils.write_tensor_to_file(weights, path + "_weights")

    gradient = utils.create_tensor(np.random.uniform(-10, 10, size=weights.shape))
    utils.write_tensor_to_file(gradient, path + "_gradient")

    utils.write_tensor_to_file(weights, path + "_updated_weights")


def sgd_optimizer():
    """
    Generates random data for testing the SGD optimizer.
    """
    path = "optimizers/sgd"

    weights = utils.create_tensor(np.random.uniform(-10, 10, size=[99, 13]))
    utils.write_tensor_to_file(weights, path + "_weights")

    gradient = utils.create_tensor(np.random.uniform(-10, 10, size=weights.shape))
    utils.write_tensor_to_file(gradient, path + "_gradient")

    learning_rate = 1e-3
    utils.write_tensor_to_file(learning_rate, path + "_learning_rate")

    optimizer = torch.optim.SGD([weights], lr=learning_rate)
    weights.grad = gradient
    optimizer.step()
    utils.write_tensor_to_file(weights, path + "_updated_weights")


def adam_optimizer():
    """
    Generates random data for testing the Adam optimizer.
    """
    path = "optimizers/adam"

    weights = utils.create_tensor(np.random.uniform(-10, 10, size=[99, 13]))
    utils.write_tensor_to_file(weights, path + "_weights")

    gradient_step_1 = utils.create_tensor(
        np.random.uniform(-10, 10, size=weights.shape)
    )
    utils.write_tensor_to_file(gradient_step_1, path + "_gradient_step_1")

    learning_rate = 5**-0.5
    utils.write_tensor_to_file(learning_rate, path + "_learning_rate")

    mu = 2**-0.5
    utils.write_tensor_to_file(mu, path + "_mu")

    rho = 3**-0.5
    utils.write_tensor_to_file(rho, path + "_rho")

    optimizer = torch.optim.Adam([weights], lr=learning_rate, betas=(mu, rho))
    weights.grad = gradient_step_1
    optimizer.step()
    utils.write_tensor_to_file(weights, path + "_updated_weights_step_1")

    first_momentum_step_1 = optimizer.state[weights]["exp_avg"]
    utils.write_tensor_to_file(first_momentum_step_1, path + "_first_momentum_step_1")

    second_momentum_step_1 = optimizer.state[weights]["exp_avg_sq"]
    utils.write_tensor_to_file(second_momentum_step_1, path + "_second_momentum_step_1")

    gradient_step_2 = utils.create_tensor(np.random.uniform(-10, 10, size=[99, 13]))
    utils.write_tensor_to_file(gradient_step_2, path + "_gradient_step_2")

    weights.grad = gradient_step_2
    optimizer.step()
    utils.write_tensor_to_file(weights, path + "_updated_weights_step_2")

    first_momentum_step_2 = optimizer.state[weights]["exp_avg"]
    utils.write_tensor_to_file(first_momentum_step_2, path + "_first_momentum_step_2")

    second_momentum_step_2 = optimizer.state[weights]["exp_avg_sq"]
    utils.write_tensor_to_file(second_momentum_step_2, path + "_second_momentum_step_2")


def noam_optimizer():
    """
    Generates random data for testing the Noam optimizer and saves the
    calculated learning rate for different number of steps.
    """
    path = "optimizers/noam"

    scale_factor = 1
    model_dim = 10
    utils.write_tensor_to_file(model_dim, path + "_model_dim")

    warmup_steps = 20
    utils.write_tensor_to_file(warmup_steps, path + "_warmup_steps")

    def noam(steps_num):
        """
        Source:
        https://github.com/AmritanshuV/Numpy-Transformer/blob/42dd602e9a651d77ddb041790b1d6d513d8b7cf7/transformer/optimizers.py#L274
        Copyright (c) 2024 AmritanshuV
        """
        return scale_factor * (
            model_dim ** (-0.5)
            * min(steps_num ** (-0.5), steps_num * warmup_steps ** (-1.5))
        )

    step = 1
    num_test = 1
    utils.write_tensor_to_file(step, f"{path}_step_{num_test}")
    utils.write_tensor_to_file(noam(step), f"{path}_learning_rate_{num_test}")

    weights = utils.create_tensor(np.random.uniform(-10, 10, size=[99, 13]))
    utils.write_tensor_to_file(weights, path + "_weights")

    gradient = utils.create_tensor(np.random.uniform(-10, 10, size=weights.shape))
    utils.write_tensor_to_file(gradient, path + "_gradient")

    optimizer = torch.optim.SGD([weights], lr=noam(step))
    weights.grad = gradient
    optimizer.step()
    utils.write_tensor_to_file(weights, path + "_updated_weights")

    step = int(warmup_steps / 2)
    num_test += 1
    utils.write_tensor_to_file(step, f"{path}_step_{num_test}")
    utils.write_tensor_to_file(noam(step), f"{path}_learning_rate_{num_test}")

    step = warmup_steps
    num_test += 1
    utils.write_tensor_to_file(step, f"{path}_step_{num_test}")
    utils.write_tensor_to_file(noam(step), f"{path}_learning_rate_{num_test}")

    step = 2 * warmup_steps
    num_test += 1
    utils.write_tensor_to_file(step, f"{path}_step_{num_test}")
    utils.write_tensor_to_file(noam(step), f"{path}_learning_rate_{num_test}")

    utils.write_tensor_to_file(num_test, path + "_num_tests")


if __name__ == "__main__":
    none_optimizer()
    sgd_optimizer()
    adam_optimizer()
    noam_optimizer()
