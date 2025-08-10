/**
 * @file matrix_multiplication.cu
 * @brief This file examines different matrix multiplication implementations and
 * measures their performance.
 *
 * The following resource helped creating this file:
 * https://siboehm.com/articles/22/CUDA-MMM
 */

#include <iostream>

#include "../src/core.cuh"
#include "../src/optimizers.cuh"
#include "../src/test.cuh"
#include "../src/utils.cuh"

/**
 * @brief This kernel multiplies `matrix_a` with `matrix_b` and saves the result
 * in `matrix_c` in a simple manor.
 *
 * @param matrix_a The first input matrix. It has to have the shape
 * @f$ [L, M] @f$.
 * @param matrix_b The second input matrix. It has to have the shape
 * @f$ [M, N] @f$.
 * @param matrix_c The output matrix. It has to have the shape @f$ [L, N] @f$.
 */
template <core::float_type data_type>
__global__ void simple_matmul_kernel(
    const core::Kernel_Tensor<data_type> matrix_a,
    const core::Kernel_Tensor<data_type> matrix_b,
    core::Kernel_Tensor<data_type> matrix_c) {
    const core::index_type row = blockIdx.x * blockDim.x + threadIdx.x;
    const core::index_type row_stride = gridDim.x * blockDim.x;
    const core::index_type col = blockIdx.y * blockDim.y + threadIdx.y;
    const core::index_type col_stride = gridDim.y * blockDim.y;

    for (core::index_type r = row; r < matrix_a.shape(0); r += row_stride) {
        for (core::index_type c = col; c < matrix_b.shape(1); c += col_stride) {
            data_type val = 0;
            for (core::index_type idx = 0; idx < matrix_a.shape(1); ++idx)
                val += matrix_a[r * matrix_a.shape(1) + idx] *
                       matrix_b[idx * matrix_b.shape(1) + c];
            matrix_c[r * matrix_b.shape(1) + c] = val;
        }
    }
}

/**
 * @brief This kernel multiplies `matrix_a` with `matrix_b` and saves the result
 * in `matrix_c` using shared memory.
 *
 * @attention It is assumed that this kernel is launched with the following
 * number of threads: `dim3(blocksize * blocksize, 1, 1)`. The number of blocks
 * can be `dim3(blocks_x, blocks_y, 1)` with `blocks_x >= 1` and
 * `blocks_y >= 1`. `blocks_x` and `blocks_y` can be calculated as usual, i.e.,
 * `int blocks_x = (N + blocksize - 1) / blocksize;` and
 * `int blocks_y = (M + blocksize - 1) / blocksize;`
 *
 * @param matrix_a The first input matrix. It has to have the shape
 * @f$ [L, M] @f$.
 * @param matrix_b The second input matrix. It has to have the shape
 * @f$ [M, N] @f$.
 * @param matrix_c The output matrix. It has to have the shape @f$ [L, N] @f$.
 *
 * @tparam blocksize The size of the block, which determines the dimensions of
 * the shared memory and the block of calculation (`blocksize` x `blocksize`).
 * @tparam data_type The data type of the matrices.
 */
template <const int blocksize, core::float_type data_type>
__global__ void shared_matmul_kernel(
    const core::Kernel_Tensor<data_type> matrix_a,
    const core::Kernel_Tensor<data_type> matrix_b,
    core::Kernel_Tensor<data_type> matrix_c) {
    // The output block that we want to compute in this thread block
    const core::index_type row_block = threadIdx.x / blocksize;
    const core::index_type col_block = threadIdx.x % blocksize;

    const core::index_type row = blockIdx.x * blocksize + row_block;
    const core::index_type col = blockIdx.y * blocksize + col_block;

    const core::index_type row_stride = gridDim.x * blocksize;
    const core::index_type col_stride = gridDim.y * blocksize;

    __shared__ data_type shared_a[blocksize * blocksize];
    __shared__ data_type shared_b[blocksize * blocksize];

    // Additional threads ("+ blocksize - matrix_a.shape(0) % blocksize") needed
    // to fill the shared matrices
    for (core::index_type r = row;
         r < matrix_a.shape(0) + blocksize - matrix_a.shape(0) % blocksize;
         r += row_stride) {
        for (core::index_type c = col;
             c < matrix_b.shape(1) + blocksize - matrix_b.shape(1) % blocksize;
             c += col_stride) {
            data_type val = 0;

            for (core::index_type shift = 0; shift < matrix_a.shape(1);
                 shift += blocksize) {
                // Copy small chunks from matrix_a and matrix_b into shared
                // memory
                core::index_type col_a = col_block  // Column inside block
                                         + shift;  // Shift the block left/right
                core::index_type row_b = row_block  // Row inside block
                                         + shift;   // Shift the block up/down

                shared_a[row_block * blocksize + col_block] =
                    r < matrix_a.shape(0) && col_a < matrix_a.shape(1)
                        ? matrix_a[r * matrix_a.shape(1) + col_a]
                        : 0;
                shared_b[row_block * blocksize + col_block] =
                    row_b < matrix_b.shape(0) && c < matrix_b.shape(1)
                        ? matrix_b[row_b * matrix_b.shape(1) + c]
                        : 0;

                __syncthreads();

                // "Matrix multiplication inside the block"
                for (core::index_type idx = 0;
                     idx < blocksize && idx + shift < matrix_a.shape(1); ++idx)
                    val += shared_a[row_block * blocksize + idx] *
                           shared_b[idx * blocksize + col_block];

                __syncthreads();
            }

            if (r < matrix_c.shape(0) && c < matrix_c.shape(1))
                matrix_c[r * matrix_c.shape(1) + c] = val;
        }
    }
}

/**
 * @brief Main function to test different matrix multiplication implementations.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument will be ignored
 * because it is assumed to be the program name. The second should be either the
 * path to a config file or the parameters required by this file. For more
 * information on the config file, see `utils::Config`.
 *
 * Example usage:
 *     `./matrix_multiplication L=500 M=500 N=500 iterations=99 warmup_steps=10`
 */
int main(int argc, char *argv[]) {
    // Preparation
    utils::Config<core::data_type> config(argc, argv);

    core::Device device;

    const core::index_type L = config.L;
    const core::index_type M = config.M;
    const core::index_type N = config.N;
    const core::index_type iterations = config.iterations;
    const core::index_type warmup_steps = config.warmup_steps;

    std::cout << std::boolalpha;
    std::cout << "L: " << L << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "iterations: " << iterations << std::endl;
    std::cout << "warmup_steps: " << warmup_steps << std::endl;
    std::cout << std::endl;

    core::Device_Tensor<core::data_type> matrix_a({L, M});
    core::Device_Tensor<core::data_type> matrix_b({M, N});
    core::Device_Tensor<core::data_type> matrix_c_1({L, N});
    core::Device_Tensor<core::data_type> matrix_c_2({L, N});

    core::checkCuda(
        core::generate_uniform(device.curand_generator(), matrix_a));
    core::checkCuda(
        core::generate_uniform(device.curand_generator(), matrix_b));

    // Benchmark
    utils::bench(
        "Matrix Multiplication - Simple",
        [&] {
            simple_matmul_kernel<<<device.blocks_2D(L, N),
                                   device.threads_2D()>>>(
                core::Kernel_Tensor(matrix_a), core::Kernel_Tensor(matrix_b),
                core::Kernel_Tensor(matrix_c_1));
        },
        iterations, warmup_steps);

    std::cout << std::endl;

    core::checkCuda(
        cublasSetMathMode(device.cublas_handle(), CUBLAS_DEFAULT_MATH));

    utils::bench(
        "Matrix Multiplication  - cuBLAS (default)",
        [&] {
            core::matrix_multiplication<core::data_type>(
                device.cublas_handle(), 1, matrix_a, M, CUBLAS_OP_N, matrix_b,
                N, CUBLAS_OP_N, 0, matrix_c_2, L, M, N);
        },
        iterations, warmup_steps);

    std::cout << "Update Correct: "
              << test::compare<core::data_type>(matrix_c_1, matrix_c_2, 1000)
              << "\n"
              << std::endl;

    core::checkCuda(
        core::generate_uniform(device.curand_generator(), matrix_c_2));

    core::checkCuda(
        cublasSetMathMode(device.cublas_handle(), CUBLAS_TF32_TENSOR_OP_MATH));

    utils::bench(
        "Matrix Multiplication  - cuBLAS (tensor)",
        [&] {
            core::matrix_multiplication<core::data_type>(
                device.cublas_handle(), 1, matrix_a, M, CUBLAS_OP_N, matrix_b,
                N, CUBLAS_OP_N, 0, matrix_c_2, L, M, N);
        },
        iterations, warmup_steps);

    std::cout << "Update Correct: "
              << test::compare<core::data_type>(matrix_c_1, matrix_c_2, 1000)
              << "\n"
              << std::endl;

    core::checkCuda(
        core::generate_uniform(device.curand_generator(), matrix_c_2));

    constexpr int blocksize = 16;
    constexpr int threads = blocksize * blocksize;
    int blocks_x = (N + blocksize - 1) / blocksize;
    int blocks_y = (M + blocksize - 1) / blocksize;
    blocks_x =
        blocks_x < device.blocks_2D().x ? blocks_x : device.blocks_2D().x;
    blocks_y =
        blocks_y < device.blocks_2D().y ? blocks_y : device.blocks_2D().y;
    dim3 blocks = {blocks_x, blocks_y, 1};

    utils::bench(
        "Matrix Multiplication  - shared",
        [&] {
            shared_matmul_kernel<blocksize, core::data_type>
                <<<blocks, threads>>>(core::Kernel_Tensor(matrix_a),
                                      core::Kernel_Tensor(matrix_b),
                                      core::Kernel_Tensor(matrix_c_2));
        },
        iterations, warmup_steps);

    std::cout << "Update Correct: "
              << test::compare<core::data_type>(matrix_c_1, matrix_c_2, 1000)
              << std::endl;
}
