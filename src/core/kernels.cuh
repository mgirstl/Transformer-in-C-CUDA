/**
 * @file kernels.cuh
 * @brief This file implements often used kernel functions.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */

#pragma once

#include "kernel_tensor.cuh"
#include "types.cuh"

/**
 * @internal
 *
 * @namespace core
 * @brief Provides core facilities used throughout the entire codebase.
 *
 * The `core` namespace contains fundamental types, utilities, and functions
 * that are widely used across different modules of the project.
 *
 * This specific file defines often used kernel functions.
 *
 * @endinternal
 */
namespace core {

    /**
     * @brief This kernel multiplies `tensor_a` and `tensor_b` element-wise and
     * saves the result in `output`.
     *
     * @param tensor_a The first input tensor.
     * @param tensor_b The second input tensor.
     * @param output The output tensor where the results will be stored in.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <arithmetic_type data_type>
    __global__ void multiply(const Kernel_Tensor<data_type> tensor_a,
                             const Kernel_Tensor<data_type> tensor_b,
                             Kernel_Tensor<data_type> output) {
        index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const index_type stride = gridDim.x * blockDim.x;

        for (; idx < output.size(); idx += stride) {
            output[idx] = tensor_a[idx] * tensor_b[idx];
        }
    }

    /**
     * @brief This kernel adds `tensor_a` and `tensor_b` element-wise and saves
     * the result in `output`.
     *
     * @param tensor_a The first input tensor.
     * @param tensor_b The second input tensor.
     * @param output The output tensor where the results will be stored in.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <arithmetic_type data_type>
    __global__ void add(const Kernel_Tensor<data_type> tensor_a,
                        const Kernel_Tensor<data_type> tensor_b,
                        Kernel_Tensor<data_type> output) {
        index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const index_type stride = gridDim.x * blockDim.x;

        for (; idx < output.size(); idx += stride) {
            output[idx] = tensor_a[idx] + tensor_b[idx];
        }
    }

    /**
     * @brief This kernel adds `tensor_a`, `tensor_b`, `tensor_c`, and
     * `tensor_d` element-wise and saves the result in `output`.
     *
     * @param tensor_a The first input tensor.
     * @param tensor_b The second input tensor.
     * @param tensor_c The third input tensor.
     * @param tensor_d The fourth input tensor.
     * @param output The output tensor where the results will be stored in.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <arithmetic_type data_type>
    __global__ void add(const Kernel_Tensor<data_type> tensor_a,
                        const Kernel_Tensor<data_type> tensor_b,
                        const Kernel_Tensor<data_type> tensor_c,
                        const Kernel_Tensor<data_type> tensor_d,
                        Kernel_Tensor<data_type> output) {
        index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const index_type stride = gridDim.x * blockDim.x;

        for (; idx < output.size(); idx += stride) {
            output[idx] =
                tensor_a[idx] + tensor_b[idx] + tensor_c[idx] + tensor_d[idx];
        }
    }

    /**
     * @brief This kernel sets all values of the input `tensor` to 1.
     */
    template <arithmetic_type data_type>
    __global__ void set_one(Kernel_Tensor<data_type> tensor) {
        index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const index_type stride = gridDim.x * blockDim.x;

        for (; idx < tensor.size(); idx += stride) {
            tensor[idx] = 1;
        }
    }

}  // namespace core
