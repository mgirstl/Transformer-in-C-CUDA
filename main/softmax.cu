/**
 * @file softmax.cu
 * @brief This file examines different softmax implementations.
 */

#include "../src/activations/softmax.cuh"
#include "../src/core.cuh"
#include "../src/test.cuh"
#include "../src/utils.cuh"

/**
 * @brief This kernel calculates the forward pass of the softmax function in a
 * single kernel.
 *
 * @param input The input tensor containing the values to be processed. It has
 * to have the shape @f$ [*, N] @f$.
 * @param output The output tensor where the results will be stored in. It has
 * to have the shape @f$ [*, N] @f$.
 *
 * @note The @f$ * @f$ symbol denotes any number of dimensions and must be the
 * same in all tensors where it is used.
 */
template <core::float_type data_type>
__global__ void simple_softmax_forward(
    const core::Kernel_Tensor<data_type> input,
    core::Kernel_Tensor<data_type> output) {
    core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
    const core::index_type stride = gridDim.x * blockDim.x;

    const core::index_type size_last_dim = input.shape(input.rank() - 1);
    const core::index_type element_ratio = input.size() / size_last_dim;

    constexpr data_type epsilon = std::numeric_limits<data_type>::epsilon() *
                                  std::numeric_limits<data_type>::epsilon();

    for (; idx < element_ratio; idx += stride) {
        const core::index_type root_idx = idx * size_last_dim;

        data_type max = 0;
        for (core::index_type j = root_idx; j < root_idx + size_last_dim; ++j)
            if (input[j] > max) max = input[j];

        data_type sum = 0;
        for (core::index_type j = root_idx; j < root_idx + size_last_dim; ++j) {
            data_type value = input[j];
            value = std::exp(value - max);
            output[j] = value;
            sum += value;
        }

        for (core::index_type j = root_idx; j < root_idx + size_last_dim; ++j) {
            if (sum < epsilon)
                output[j] = 1. / size_last_dim;
            else
                output[j] /= sum;
        }
    }
}

/**
 * @brief This kernel calculates the maximum along the last axis of a tensor.
 *
 * @param input The input tensor containing the values to be processed. It has
 * to have the shape @f$ [*, N] @f$.
 * @param max The output tensor to store the maximum along the last axis in. It
 * has to have the shape @f$ [*] @f$.
 *
 * @note The @f$ * @f$ symbol denotes any number of dimensions and must be the
 * same in all tensors where it is used.
 */
template <core::float_type data_type>
__global__ void simple_softmax_forward_max(
    const core::Kernel_Tensor<data_type> input,
    core::Kernel_Tensor<data_type> max) {
    core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
    const core::index_type stride = gridDim.x * blockDim.x;

    const core::index_type size_last_dim = input.shape(input.rank() - 1);

    for (; idx < max.size(); idx += stride) {
        const core::index_type root_idx = idx * size_last_dim;

        data_type _max = -std::numeric_limits<data_type>::max();
        for (core::index_type j = root_idx; j < root_idx + size_last_dim; ++j)
            if (input[j] > _max) _max = input[j];
        max[idx] = _max;
    }
}

/**
 * @brief Main function to test different softmax implementations.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument will be ignored
 * because it is assumed to be the program name. The second should be either the
 * path to a config file or the parameters required by this file. For more
 * information on the config file, see `utils::Config`.
 *
 * Example usage:
 *     `./softmax M=10000 N=100000 iterations=100 warmup_steps=10`
 */
int main(int argc, char *argv[]) {
    // Preparation
    utils::Config<core::data_type> config(argc, argv);

    core::Device device;

    const core::index_type M = config.M;
    const core::index_type N = config.N;
    const core::index_type iterations = config.iterations;
    const core::index_type warmup_steps = config.warmup_steps;

    std::cout << std::boolalpha;
    std::cout << "M: " << M << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "iterations: " << iterations << std::endl;
    std::cout << "warmup_steps: " << warmup_steps << std::endl;
    std::cout << std::endl;

    core::Device_Tensor<core::data_type> input({M, N});
    core::Device_Tensor<core::data_type> output_1({M, N});
    core::Device_Tensor<core::data_type> output_2({M, N});
    core::Device_Tensor<core::data_type> one({N});
    core::Device_Tensor<core::data_type> stat({M});

    core::checkCuda(core::generate_uniform(device.curand_generator(), input));

    // Benchmark
    utils::bench(
        "Softmax - Single Kernel",
        [&] {
            simple_softmax_forward<<<device.blocks(M), device.threads()>>>(
                core::Kernel_Tensor(input), core::Kernel_Tensor(output_1));
        },
        iterations, warmup_steps);

    std::cout << std::endl;

    core::checkCuda(
        cublasSetMathMode(device.cublas_handle(), CUBLAS_DEFAULT_MATH));
    utils::bench(
        "Softmax - Simple Max (default)",
        [&] {
            simple_softmax_forward_max<<<device.blocks(M), device.threads()>>>(
                core::Kernel_Tensor(input), core::Kernel_Tensor(stat));
            core::checkCuda(cudaPeekAtLastError());

            activations::
                softmax_forward_exp<<<device.blocks(M * N), device.threads()>>>(
                    core::Kernel_Tensor(input), core::Kernel_Tensor(output_2),
                    core::Kernel_Tensor(stat));
            core::checkCuda(cudaPeekAtLastError());

            core::matrix_vector_multiplication<core::data_type>(
                device.cublas_handle(), 1.0, output_2, M, N, CUBLAS_OP_N, one,
                0.0, stat);
            core::checkCuda(cudaPeekAtLastError());

            activations::softmax_forward_normalize<<<device.blocks(M * N),
                                                     device.threads()>>>(
                core::Kernel_Tensor(output_2), core::Kernel_Tensor(stat));
            core::checkCuda(cudaPeekAtLastError());
        },
        iterations, warmup_steps);

    std::cout << "Update Correct: "
              << test::compare<core::data_type>(output_1, output_2, 1000)
              << "\n"
              << std::endl;

    core::checkCuda(
        core::generate_uniform(device.curand_generator(), output_2));

    core::checkCuda(
        cublasSetMathMode(device.cublas_handle(), CUBLAS_TF32_TENSOR_OP_MATH));

    utils::bench(
        "Softmax - Simple Max (tensor)",
        [&] {
            simple_softmax_forward_max<<<device.blocks(M), device.threads()>>>(
                core::Kernel_Tensor(input), core::Kernel_Tensor(stat));
            core::checkCuda(cudaPeekAtLastError());

            activations::
                softmax_forward_exp<<<device.blocks(M * N), device.threads()>>>(
                    core::Kernel_Tensor(input), core::Kernel_Tensor(output_2),
                    core::Kernel_Tensor(stat));
            core::checkCuda(cudaPeekAtLastError());

            core::matrix_vector_multiplication<core::data_type>(
                device.cublas_handle(), 1.0, output_2, M, N, CUBLAS_OP_N, one,
                0.0, stat);
            core::checkCuda(cudaPeekAtLastError());

            activations::softmax_forward_normalize<<<device.blocks(M * N),
                                                     device.threads()>>>(
                core::Kernel_Tensor(output_2), core::Kernel_Tensor(stat));
            core::checkCuda(cudaPeekAtLastError());
        },
        iterations, warmup_steps);

    std::cout << "Update Correct: "
              << test::compare<core::data_type>(output_1, output_2, 1000)
              << "\n"
              << std::endl;

    activations::Softmax<core::data_type> softmax{device, {N}};

    core::checkCuda(
        cublasSetMathMode(device.cublas_handle(), CUBLAS_DEFAULT_MATH));

    utils::bench(
        "Softmax (default)", [&] { softmax.forward(input); }, iterations,
        warmup_steps);

    std::cout << "Update Correct: "
              << test::compare<core::data_type>(output_1, softmax.get_output(),
                                                1000)
              << "\n"
              << std::endl;

    core::checkCuda(
        core::generate_uniform(device.curand_generator(), output_2));

    core::checkCuda(
        cublasSetMathMode(device.cublas_handle(), CUBLAS_TF32_TENSOR_OP_MATH));

    utils::bench(
        "Softmax (tensor)", [&] { softmax.forward(input); }, iterations,
        warmup_steps);

    std::cout << "Update Correct: "
              << test::compare<core::data_type>(output_1, softmax.get_output(),
                                                1000)
              << "\n"
              << std::endl;

    utils::bench(
        "Simple Max",
        [&] {
            simple_softmax_forward_max<<<device.blocks(M), device.threads()>>>(
                core::Kernel_Tensor(input), core::Kernel_Tensor(stat));
        },
        iterations, warmup_steps);

    std::cout << std::endl;

    // Copied `from src/activations/softmax.cuh`:
        // Calculate the optimal number of blocks and threads for the
        // softmax_forward_max kernel:
        dim3 threads = {1, 1, 1};

        // Upper limit for threads.y (threads.y needs to be divisible by 2).
        threads.y = (N + 1) / 2;
        threads.y =
            threads.y < device.max_threads() ? threads.y : device.max_threads();
        threads.y = threads.y % 2 ? threads.y - 1 : threads.y;
        threads.y = threads.y ? threads.y : 2;

        // Lower limit for threads.x
        threads.x = device.max_threads() / threads.y;

        // Number of blocks according to the lower limit of threads.x.
        core::index_type blocks = (stat.size() + threads.x - 1) / threads.x;

        // If we have too many blocks:
        // - Increase threads.x
        // - Decrease threads.y
        // Reason: The softmax_forward_max kernel applies reduction along
        // the axis covered by threads.y. Hence, over time, more and more
        // threads along this axis become idle.
        if (blocks > device.blocks()) {
            blocks = device.blocks();
            threads.x = (stat.size() + blocks - 1) / blocks;
            threads.x = threads.x < device.max_threads()
                      ? threads.x : device.max_threads();
            threads.y = device.max_threads() / threads.x;
        }

        threads.y = threads.y % 2 ? threads.y - 1 : threads.y;
        threads.y = threads.y ? threads.y : 2;
        threads.x = device.max_threads() / threads.y;

    utils::bench(
        "Reduction Max",
        [&] {
            activations::softmax_forward_max<<<blocks, threads,
                threads.x * threads.y * sizeof(core::data_type)>>>(
                core::Kernel_Tensor(input), core::Kernel_Tensor(stat));
        },
        iterations, warmup_steps);
}
