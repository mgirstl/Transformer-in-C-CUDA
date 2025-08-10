/**
 * @file indicator.cuh
 * @brief Implements the Indicator class.
 *
 * This file is part of the `utils` namespace, which implements utility
 * functions and classes.
 */

#pragma once

#include <vector>

#include "../core.cuh"

/**
 * @internal
 *
 * @namespace utils
 * @brief Namespace for utility functions and classes.
 *
 * This specific file defines the Indicator class.
 *
 * @endinternal
 */
namespace utils {

    /**
     * @brief This kernel applies the Indicator function.
     *
     * This kernel applies the Indicator function element-wise to the input and
     * target tensor.
     *
     * @param input The input tensor containing the values to compare.
     * @param output The output tensor to save the result of the comparison in.
     * @param target The target tensor to compare with.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <core::arithmetic_type data_type>
    __global__ void indicator_apply(
        const core::Kernel_Tensor<data_type> input,
        core::Kernel_Tensor<core::mask_type> output,
        const core::Kernel_Tensor<data_type> target) {
        core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
        core::index_type stride = gridDim.x * blockDim.x;

        for (; idx < input.size(); idx += stride)
            output[idx] = input[idx] == target[idx];
    }

    /**
     * @class Indicator
     * @brief The Indicator function.
     *
     * The Indicator function is defined as:
     * @f[
     *     f(x_j, y_j) = \left\{ \begin{array}{ll}
     *                       1, & x_j = y_j \\
     *                       0, & x_j \neq y_j
     *                   \end{array} \right.
     * @f]
     * for each element @f$ x_j @f$ and @f$ y_j @f$ of the tensor @f$ x @f$ and
     * @f$ y @f$ respectively.
     */
    template <core::arithmetic_type data_type>
    class Indicator {
      private:
        const core::Device &device;
        core::Device_Tensor<core::mask_type> output;

      public:
        /**
         * @brief Construct a new Indicator object
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape (excluding the first axis) of the input/output
         * tensors.
         */
        Indicator(const core::Device &device,
                  const std::vector<core::index_type> &shape)
            : device{device}, output{core::NO_ALLOC, shape} {}

        /**
         * @brief Compute the Indicator function.
         *
         * @param input The input tensor containing the values to compare.
         * @param target The target tensor to compare with.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<core::mask_type> &apply(
            const core::Device_Tensor<data_type> &input,
            const core::Device_Tensor<data_type> &target,
            const cudaStream_t stream = core::default_stream) {
            output.rebatchsize(input.batchsize(), stream);

            indicator_apply<data_type>
                <<<device.blocks(output.size()), device.threads(), 0, stream>>>(
                    input, output, target);
            core::checkCuda(cudaPeekAtLastError());

            return output;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(const bool /* unused */ = false) const {
            return output.mem_size();
        }
    };

}  // namespace utils
