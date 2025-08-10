/**
 * @file argmax.cuh
 * @brief Implements the Argmax class.
 *
 * This file is part of the `utils` namespace, which implements utility
 * functions and classes.
 */

#pragma once

#include <limits>
#include <vector>

#include "../core.cuh"

/**
 * @internal
 *
 * @namespace utils
 * @brief Namespace for utility functions and classes.
 *
 * This specific file defines the Argmax class.
 *
 * @endinternal
 */
namespace utils {

    /**
     * @brief This kernel applies the Argmax function along the last axis of the
     * input tensor.
     *
     * @note If this kernel is frequently used, the internal loop that
     * calculates the maximum can be optimized using reduction techniques. For
     * example, see `activations::softmax_forward_max`.
     *
     * @param input The input tensor containing the values to be processed. It
     * must have the shape @f$ [*, N] @f$.
     * @param output The index of the largest element of each input sample. It
     * must have the shape @f$ [*] @f$.
     * @param ignore_index Specifies a index to ignore during the maximum
     * calculation.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::arithmetic_type input_type,
              core::arithmetic_type label_type>
    __global__ void argmax_apply(const core::Kernel_Tensor<input_type> input,
                                 core::Kernel_Tensor<label_type> output,
                                 const label_type ignore_index) {
        core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        const core::index_type size_last_dim = input.shape(input.rank() - 1);
        const core::index_type element_ratio = input.size() / size_last_dim;

        for (; idx < element_ratio; idx += stride) {
            const core::index_type root_idx = idx * size_last_dim;

            input_type max = -std::numeric_limits<input_type>::max();
            label_type argmax = 0;
            for (core::index_type j = 0; j < size_last_dim; ++j) {
                if (j == ignore_index) continue;
                if (input[j + root_idx] > max) {
                    max = input[j + root_idx];
                    argmax = j;
                }
            }

            output[idx] = argmax;
        }
    }

    /**
     * @class Argmax
     * @brief The Argmax function along the last axis of the input.
     */
    template <core::arithmetic_type input_type,
              core::arithmetic_type label_type>
    class Argmax {
      private:
        const core::Device &device;
        core::index_type ignore_index;
        core::Device_Tensor<label_type> output;

      public:
        /**
         * @brief Construct a new Argmax object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape (excluding the first axis) of the input
         * tensors.
         * @param ignore_index Specifies a index to ignore during the maximum
         * calculation.
         */
        Argmax(const core::Device &device,
               const std::vector<core::index_type> &shape,
               const core::index_type ignore_index =
                   std::numeric_limits<core::index_type>::max())
            : device{device},
              ignore_index{ignore_index},
              output{core::NO_ALLOC, std::vector<core::index_type>(
                                         shape.begin(), shape.end() - 1)} {}

        /**
         * @brief Compute the Argmax along the last axis of the input.
         *
         * @param input The input tensor containing the values to be processed.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<label_type> &apply(
            const core::Device_Tensor<input_type> &input,
            const cudaStream_t stream = core::default_stream) {
            output.rebatchsize(input.batchsize(), stream);

            argmax_apply<input_type, label_type>
                <<<device.blocks(output.size()), device.threads(), 0, stream>>>(
                    input, output, ignore_index);
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
