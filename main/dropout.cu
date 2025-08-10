/**
 * @file dropout.cu
 * @brief This file examines different dropout implementations.
 */

#include <iostream>

#include "../src/core.cuh"
#include "../src/layers/dropout.cuh"
#include "../src/utils.cuh"

/**
 * @brief This kernel calculates the forward pass of the Dropout layer in a
 * simple manor.
 *
 * This kernel sets elements of the input tensor to zero with a certain
 * probability. Additionally, a scaling is applied so that the intensity of the
 * layer (the expected sum of the propagated values) remains consistent
 * regardless of the dropout probability.
 *
 * @param input The input tensor containing the values to be processed.
 * @param output The output tensor where the results will be stored in.
 * @param mask The mask tensor will store the mask indicating which elements
 * were not set to zero.
 * @param probability The probability of setting an element to zero.
 * @param rng_state Pointer to the array of cuRAND states. The array must be at
 * least as large as the maximum number of unique threads that will be
 * launched.
 *
 * @note It is assumed that all the tensors have the same size.
 */
template <core::arithmetic_type data_type>
__global__ void simple_dropout_forward(
    const core::Kernel_Tensor<data_type> input,
    core::Kernel_Tensor<data_type> output, core::Kernel_Tensor<data_type> mask,
    const data_type probability, curandStatePhilox4_32_10_t *rng_state) {
    const core::index_type thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const core::index_type stride = gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t _rng_state = rng_state[thread_id];
    for (core::index_type idx = thread_id; idx < input.size(); idx += stride) {
        const data_type m = core::uniform<data_type>(_rng_state) > probability;
        mask[idx] = m;
        output[idx] = m * input[idx] / (1 - probability);
    }
    rng_state[thread_id] = _rng_state;
}

/**
 * @brief This kernel calculates the forward pass of the Dropout layer using
 * pre-computed random numbers.
 *
 * This kernel sets elements of the input tensor to zero with a certain
 * probability. Additionally, a scaling is applied so that the intensity of
 * the layer (the expected sum of the propagated values) remains consistent
 * regardless of the dropout probability.
 *
 * @param input The input tensor containing the values to be processed.
 * @param output The output tensor where the results will be stored in.
 * @param mask A tensor that stores uniform random values between 0 and 1.
 * During the calculation the mask indicating which elements were not set to
 * zero will be written to it.
 * @param probability The probability of setting an element to zero.
 *
 * @note It is assumed that all the tensors have the same size.
 */
template <core::arithmetic_type data_type>
__global__ void curand_dropout_forward(
    const core::Kernel_Tensor<data_type> input,
    core::Kernel_Tensor<data_type> output, core::Kernel_Tensor<data_type> mask,
    const data_type probability) {
    core::index_type thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    core::index_type stride = gridDim.x * blockDim.x;

    for (core::index_type idx = thread_id; idx < input.size(); idx += stride) {
        data_type m = mask[idx] > probability;
        mask[idx] = m;
        output[idx] = m * input[idx] / (1 - probability);
    }
}

/**
 * @brief Main function to test different dropout implementations.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument will be ignored
 * because it is assumed to be the program name. The second should be either the
 * path to a config file or the parameters required by this file. For more
 * information on the config file, see `utils::Config`.
 *
 * Example usage:
 *     `./dropout N=99999999 iterations=100 warmup_steps=10`
 */
int main(int argc, char *argv[]) {
    // Preparation
    utils::Config<core::data_type> config(argc, argv);

    core::Device device;

    auto &rng = utils::Random_Number_Generator<
        curandStatePhilox4_32_10_t>::get_instance(device);

    core::data_type probability = 0.5;

    const core::index_type size = config.N;
    const core::index_type iterations = config.iterations;
    const core::index_type warmup_steps = config.warmup_steps;

    std::cout << "size: " << size << std::endl;
    std::cout << "iterations: " << iterations << std::endl;
    std::cout << "warmup_steps: " << warmup_steps << std::endl;
    std::cout << std::endl;

    core::Device_Tensor<core::data_type> input({size});
    core::Device_Tensor<core::data_type> output({size});
    core::Device_Tensor<core::data_type> mask({size});

    core::checkCuda(core::generate_normal(device.curand_generator(), input));

    constexpr int elements_per_vector =
        core::vector_type<core::data_type>::elements_per_vector;
    core::index_type _size =
        (size + elements_per_vector - 1) / elements_per_vector;

    // Benchmark
    utils::bench(
        "Dropout - Simple",
        [&] {
            simple_dropout_forward<<<device.blocks(size), device.threads()>>>(
                core::Kernel_Tensor(input), core::Kernel_Tensor(output),
                core::Kernel_Tensor(mask), probability, rng.states());
        },
        iterations, warmup_steps);

    std::cout << std::endl;

    utils::bench(
        "Dropout - Vectorized",
        [&] {
            layers::dropout_forward<<<device.blocks(_size), device.threads()>>>(
                core::Kernel_Tensor(input), core::Kernel_Tensor(output),
                core::Kernel_Tensor(mask), probability, rng.states());
        },
        iterations, warmup_steps);

    std::cout << std::endl;

    utils::bench(
        "Dropout - cuRAND",
        [&] {
            core::checkCuda(
                core::generate_uniform(device.curand_generator(), mask));
            curand_dropout_forward<<<device.blocks(size), device.threads()>>>(
                core::Kernel_Tensor(input), core::Kernel_Tensor(output),
                core::Kernel_Tensor(mask), probability);
        },
        iterations, warmup_steps);
}
