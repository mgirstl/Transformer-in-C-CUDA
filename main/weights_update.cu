/**
 * @file weights_update.cu
 * @brief This file examines different weights update implementations.
 */

#include <cublas_v2.h>

#include <iostream>

#include "../src/core.cuh"
#include "../src/optimizers.cuh"
#include "../src/test.cuh"
#include "../src/utils.cuh"

/**
 * @brief This kernel calculates the SGD update in a simple manor.
 *
 * @param weights The tensor which will be updated.
 * @param gradient The gradient values to add to the weights.
 * @param learning_rate The learning rate used for scaling the gradient.
 *
 * @note It is assumed that all the tensors have the same size.
 */
template <core::float_type data_type>
__global__ void simple_sgd_update(core::Kernel_Tensor<data_type> weights,
                                  const core::Kernel_Tensor<data_type> gradient,
                                  const data_type learning_rate) {
    core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
    const core::index_type stride = gridDim.x * blockDim.x;

    for (; idx < weights.size(); idx += stride)
        weights[idx] -= learning_rate * gradient[idx];
}

/**
 * @brief This kernel calculates the SGD update using cuBLAS saxpy
 * implementation.
 *
 * @param weights The tensor which will be updated.
 * @param gradient The gradient values to add to the weights.
 * @param learning_rate The learning rate used for scaling the gradient.
 *
 * @note It is assumed that all the tensors have the same size.
 */
template <core::float_type data_type>
cublasStatus_t inline cublas_sgd_update(
    core::Device &device, core::Kernel_Tensor<data_type> weights,
    const core::Kernel_Tensor<data_type> gradient,
    const data_type learning_rate) {
    const data_type neg_learning_rate = -learning_rate;
    return core::checkCuda(cublasSaxpy(device.cublas_handle(), weights.size(),
                                       &neg_learning_rate, gradient.data(), 1,
                                       weights.data(), 1));
}

/**
 * @brief This kernel calculates the Adam update in a simple manor.
 *
 * @param weights The tensor which will be updated.
 * @param gradient The gradient values to add to the weights.
 * @param alpha The learning rate used for scaling the gradient.
 * @param first_moment The first moment calculated over the previous updates.
 * @param second_moment The second moment calculated over the previous updates.
 * @param mu The scaling parameter for the first moment.
 * @param rho The scaling parameter for the second moment.
 *
 * @note It is assumed that all the tensors have the same size.
 */
template <core::float_type data_type>
__global__ void simple_adam_update(
    core::Kernel_Tensor<data_type> weights,
    const core::Kernel_Tensor<data_type> gradient, const data_type alpha,
    core::Kernel_Tensor<data_type> first_moment,
    core::Kernel_Tensor<data_type> second_moment, const data_type mu,
    const data_type rho) {
    core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    const core::index_type stride = gridDim.x * blockDim.x;

    constexpr data_type epsilon = std::numeric_limits<data_type>::epsilon() *
                                  std::numeric_limits<data_type>::epsilon();

    for (; idx < weights.size(); idx += stride) {
        const data_type g = gradient[idx];
        data_type v = first_moment[idx];
        data_type r = second_moment[idx];

        v = mu * v + (1 - mu) * g;
        r = rho * r + (1 - rho) * g * g;

        weights[idx] -= alpha * v / (sqrt(r) + epsilon);
        first_moment[idx] = v;
        second_moment[idx] = r;
    }
}

/**
 * @brief Main function to test different weights update implementations.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument will be ignored
 * because it is assumed to be the program name. The second should be either the
 * path to a config file or the parameters required by this file. For more
 * information on the config file, see `utils::Config`.
 *
 * Example usage:
 *     `./weights_update N=99999999 iterations=100 warmup_steps=10`
 */
int main(int argc, char *argv[]) {
    // Preparation
    utils::Config<core::data_type> config(argc, argv);

    core::Device device;

    core::data_type mu = 0.2;
    core::data_type rho = 0.3;
    core::data_type learning_rate = 1.;

    const core::index_type size = config.N;
    const core::index_type iterations = config.iterations;
    const core::index_type warmup_steps = config.warmup_steps;

    std::cout << std::boolalpha;
    std::cout << "size: " << size << std::endl;
    std::cout << "iterations: " << iterations << std::endl;
    std::cout << "warmup_steps: " << warmup_steps << std::endl;
    std::cout << std::endl;

    core::Device_Tensor<core::data_type> weights({size});
    core::Device_Tensor<core::data_type> weights_1;
    core::Device_Tensor<core::data_type> weights_2;
    core::Device_Tensor<core::data_type> gradient({size});
    core::Device_Tensor<core::data_type> first_moment({size});
    core::Device_Tensor<core::data_type> first_moment_1;
    core::Device_Tensor<core::data_type> first_moment_2;
    core::Device_Tensor<core::data_type> second_moment({size});
    core::Device_Tensor<core::data_type> second_moment_1;
    core::Device_Tensor<core::data_type> second_moment_2;

    core::checkCuda(core::generate_uniform(device.curand_generator(), weights));
    core::checkCuda(
        core::generate_uniform(device.curand_generator(), gradient));
    core::checkCuda(
        core::generate_uniform(device.curand_generator(), first_moment));
    core::checkCuda(
        core::generate_uniform(device.curand_generator(), second_moment));

    constexpr int elements_per_vector =
        core::vector_type<core::data_type>::elements_per_vector;
    core::index_type _size =
        (size + elements_per_vector - 1) / elements_per_vector;

    // Benchmark
    weights_1 = weights;
    utils::bench(
        "SGD - Simple",
        [&] {
            simple_sgd_update<<<device.blocks(size), device.threads()>>>(
                core::Kernel_Tensor(weights_1), core::Kernel_Tensor(gradient),
                learning_rate);
        },
        iterations, warmup_steps);

    std::cout << std::endl;

    weights_2 = weights;
    utils::bench(
        "SGD - Vectorized",
        [&] {
            optimizers::sgd_update<<<device.blocks(_size), device.threads()>>>(
                core::Kernel_Tensor(weights_2), core::Kernel_Tensor(gradient),
                learning_rate);
        },
        iterations, warmup_steps);

    std::cout << "Update Correct: " << test::compare(weights_1, weights_2)
              << "\n"
              << std::endl;

    core::checkCuda(
        cublasSetMathMode(device.cublas_handle(), CUBLAS_DEFAULT_MATH));

    weights_2 = weights;
    utils::bench(
        "SGD - cuBLAS (default)",
        [&] {
            cublas_sgd_update(device, core::Kernel_Tensor(weights_2),
                              core::Kernel_Tensor(gradient), learning_rate);
        },
        iterations, warmup_steps);

    std::cout << "Update Correct: " << test::compare(weights_1, weights_2)
              << "\n"
              << std::endl;

    core::checkCuda(
        cublasSetMathMode(device.cublas_handle(), CUBLAS_TF32_TENSOR_OP_MATH));

    weights_2 = weights;
    utils::bench(
        "SGD - cuBLAS (tensor)",
        [&] {
            cublas_sgd_update(device, core::Kernel_Tensor(weights_2),
                              core::Kernel_Tensor(gradient), learning_rate);
        },
        iterations, warmup_steps);

    std::cout << "Update Correct: " << test::compare(weights_1, weights_2)
              << "\n"
              << std::endl;

    weights_1 = weights;
    first_moment_1 = first_moment;
    second_moment_1 = second_moment;
    utils::bench(
        "Adam - Simple",
        [&] {
            simple_adam_update<<<device.blocks(size), device.threads()>>>(
                core::Kernel_Tensor(weights_1), core::Kernel_Tensor(gradient),
                learning_rate, core::Kernel_Tensor(first_moment_1),
                core::Kernel_Tensor(second_moment_1), mu, rho);
        },
        iterations, warmup_steps);

    std::cout << std::endl;

    weights_2 = weights;
    first_moment_2 = first_moment;
    second_moment_2 = second_moment;
    utils::bench(
        "Adam - Vectorized",
        [&] {
            optimizers::adam_update<<<device.blocks(_size), device.threads()>>>(
                core::Kernel_Tensor(weights_2), core::Kernel_Tensor(gradient),
                learning_rate, core::Kernel_Tensor(first_moment_2),
                core::Kernel_Tensor(second_moment_2), mu, rho);
        },
        iterations, warmup_steps);

    std::cout << "Update Correct - weights: "
              << test::compare<core::data_type>(weights_1, weights_2, 1000)
              << std::endl;
    std::cout << "Update Correct - first moment: "
              << test::compare(first_moment_1, first_moment_2) << std::endl;
    std::cout << "Update Correct - second moment: "
              << test::compare(second_moment_1, second_moment_2) << std::endl;
    std::cout << std::endl;
}
