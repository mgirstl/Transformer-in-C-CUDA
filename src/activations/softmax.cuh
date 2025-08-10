/**
 * @file softmax.cuh
 * @brief Implements the Softmax function.
 *
 * This file is part of the `activations` namespace, which implements
 * differentiable activation functions.
 */

#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "../core.cuh"
#include "activation.cuh"

/**
 * @internal
 *
 * @namespace activations
 * @brief Implements differentiable activation functions.
 *
 * This specific file defines the ReLU function.
 *
 * @endinternal
 */
namespace activations {

    /**
     * @brief This kernel calculates the maximum operation along the last axis
     * of the input tensor.
     *
     * @attention When launching this kernel, the number of threads in the
     * y-direction (`blockDim.y`) needs to be divisible by 2 and the number of
     * blocks in the y-direction (`gridDim.y`) needs to be 1!
     * @attention It is assumed, then when launching this kernel, the shared
     * memory has the size `blockDim.x * blockDim.y * sizeof(data_type)`.
     *
     * @param input The input tensor containing the values to be processed. It
     * has to have the shape @f$ [*, N] @f$.
     * @param max The output tensor to store the maximum along the last axis in.
     * It has to have the shape @f$ [*] @f$.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::float_type data_type>
    __global__ void softmax_forward_max(
        const core::Kernel_Tensor<data_type> input,
        core::Kernel_Tensor<data_type> max) {
        const core::index_type row_block = threadIdx.x;
        const core::index_type col_block = threadIdx.y;
        const core::index_type blocksize = blockDim.y;

        const core::index_type row = blockIdx.x * blockDim.x + row_block;
        const core::index_type row_stride = gridDim.x * blockDim.x;

        extern __shared__ data_type shared_max[];

        const core::index_type num_cols = input.shape(input.rank() - 1);
        const core::index_type num_rows = input.size() / num_cols;

        for (core::index_type r = row; r < num_rows; r += row_stride) {
            core::index_type block_idx = row_block * blocksize + col_block;

            // Initialize shared memory
            data_type _max = -std::numeric_limits<data_type>::max();
            for (core::index_type idx = col_block; idx < num_cols;
                 idx += blocksize) {
                data_type _val = input[r * num_cols + idx];
                _max = _val > _max ? _val : _max;
            }
            shared_max[block_idx] = _max;

            __syncthreads();

            // Block reduction
            for (core::index_type stride = blocksize / 2; stride > 1;
                 stride = (stride + 1) / 2) {
                if (col_block >= stride) continue;

                data_type _val1 = shared_max[block_idx];
                data_type _val2 = shared_max[block_idx + stride];
                shared_max[block_idx] = _val1 > _val2 ? _val1 : _val2;

                __syncthreads();
            }

            // Save the result
            if (col_block == 0) {
                data_type _val1 = shared_max[row_block * blocksize];
                data_type _val2 = shared_max[row_block * blocksize + 1];
                max[r] = _val1 > _val2 ? _val1 : _val2;
            }
        }
    }

    /**
     * @brief This kernel calculates the element-wise exponential of the input
     * minus the maximum along the last axis of the input.
     *
     * @param input The input tensor containing the values to be processed. It
     * has to have the shape @f$ [*, N] @f$.
     * @param output The output tensor where the results will be stored in. It
     * has to have the shape @f$ [*, N] @f$.
     * @param max The maximum along the last axis of the input tensor. It has to
     * have the shape @f$ [*] @f$.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::float_type data_type>
    __global__ void softmax_forward_exp(
        const core::Kernel_Tensor<data_type> input,
        core::Kernel_Tensor<data_type> output,
        const core::Kernel_Tensor<data_type> max) {
        core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        const core::index_type size_last_dim = input.shape(input.rank() - 1);

        for (; idx < input.size(); idx += stride) {
            output[idx] = std::exp(input[idx] - max[idx / size_last_dim]);
        }
    }

    /**
     * @brief This kernel normalized the input using the normalization
     * calculated along the last axis of the input.
     *
     * @param output The input and output tensor containing the values to be
     * processed and where the results will be stored in. It has to have the
     * shape @f$ [*, N] @f$.
     * @param norm The normalization values along the last axis of output
     * tensor. It has to have the shape @f$ [*] @f$.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::float_type data_type>
    __global__ void softmax_forward_normalize(
        core::Kernel_Tensor<data_type> output,
        const core::Kernel_Tensor<data_type> norm) {
        core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        const core::index_type size_last_dim = output.shape(output.rank() - 1);

        constexpr data_type epsilon =
            std::numeric_limits<data_type>::epsilon() *
            std::numeric_limits<data_type>::epsilon();

        for (; idx < output.size(); idx += stride) {
            data_type _norm = norm[idx / size_last_dim];
            if (_norm < epsilon)
                output[idx] = data_type(1.) / size_last_dim;
            else
                output[idx] /= _norm;
        }
    }

    /**
     * @brief This kernel calculates the last step of the backwards propagation
     * of the Softmax.
     *
     * @param output The output tensor of the forward pass. It has to have the
     * shape @f$ [*, N] @f$.
     * @param input_gradient The tensor where the result will be saved in. It
     * has to have the shape @f$ [*, N] @f$.
     * @param sum The sum along the last axis of the tensor containing the error
     * values from the next layer multiplied with the output of the forward pass
     * of this layer. It has to have the shape @f$ [*] @f$.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::float_type data_type>
    __global__ void softmax_backward_propagate(
        const core::Kernel_Tensor<data_type> output,
        core::Kernel_Tensor<data_type> input_gradient,
        const core::Kernel_Tensor<data_type> sum) {
        core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        const core::index_type size_last_dim = output.shape(output.rank() - 1);
        for (; idx < output.size(); idx += stride) {
            input_gradient[idx] -= sum[idx / size_last_dim] * output[idx];
        }
    }

    /**
     * @class Softmax
     * @brief The Softmax function.
     *
     * The Softmax function is defined as:
     * @f[ f(x_j) = \frac{\exp{x_j}}{\sum_{k \in K(j)} \exp{x_k}} @f]
     * for each element @f$ x_j @f$ of the tensor @f$ x @f$. @f$ K(j) @f$
     * represents the set of elements along a specific axis of the tensor
     * @f$ x @f$.
     *
     * @note The Softmax implemented here operates along the last axis of the
     * tensor. For example, if @f$ x @f$ is a matrix, @f$ K(j) @f$ would be the
     * row containing @f$ x_j @f$.
     */
    template <core::float_type data_type>
    class Softmax final : public Activation<data_type> {
      private:
        core::Device &device;
        core::index_type size_last_dim;
        core::Device_Tensor<data_type> one;
        core::Device_Tensor<data_type> stat;  // multipurpose variable
        core::Device_Tensor<data_type> output;
        core::Device_Tensor<data_type> input_gradient;

      public:
        /**
         * @brief Construct a new Softmax object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape (excluding the first axis) of the input/output
         * tensors.
         * @param stream The CUDA stream used for launching the kernels.
         */
        Softmax(core::Device &device,
                const std::vector<core::index_type> &shape,
                const cudaStream_t stream = core::default_stream)
            : device{device},
              size_last_dim{shape.size() == 0 ? 1 : shape.back()},
              one{{size_last_dim}},
              stat{core::NO_ALLOC, {}},
              output{core::NO_ALLOC, shape},
              input_gradient{core::NO_ALLOC, shape} {
            core::set_one<data_type>
                <<<device.blocks(one.size()), device.threads(), 0, stream>>>(
                    one);
            core::checkCuda(cudaPeekAtLastError());
        }

        /**
         * @brief Compute the forward pass of the Softmax function.
         *
         * @param input The input tensor containing the values to be processed.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const cudaStream_t stream = core::default_stream) override {
            output.rebatchsize(input.batchsize(), stream);

            const core::index_type n = input.size() / size_last_dim;
            stat.rebatchsize(n, stream);

            // Calculate the optimal number of blocks and threads for the
            // softmax_forward_max kernel:
            dim3 threads = {1, 1, 1};

            // Upper limit for threads.y (threads.y needs to be divisible by 2).
            threads.y = (size_last_dim + 1) / 2;
            threads.y = threads.y < device.max_threads()
                      ? threads.y : device.max_threads();
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
                                ? threads.x
                                : device.max_threads();
                threads.y = device.max_threads() / threads.x;
            }

            // Apply restrictions again, i.e., threads.y needs to be divisible
            // by 2 and the total number of threads needs to be smaller than
            // device.max_threads().
            threads.y = threads.y % 2 ? threads.y - 1 : threads.y;
            threads.y = threads.y ? threads.y : 2;
            threads.x = device.max_threads() / threads.y;

            // Launch kernels
            softmax_forward_max<data_type>
                <<<blocks, threads, threads.x * threads.y * sizeof(data_type),
                   stream>>>(input, stat);
            core::checkCuda(cudaPeekAtLastError());

            softmax_forward_exp<data_type>
                <<<device.blocks(input.size()), device.threads(), 0, stream>>>(
                    input, output, stat);
            core::checkCuda(cudaPeekAtLastError());

            core::matrix_vector_multiplication<data_type>(
                device.cublas_handle(stream), 1.0, output, n, size_last_dim,
                CUBLAS_OP_N, one, 0.0, stat);
            core::checkCuda(cudaPeekAtLastError());

            softmax_forward_normalize<data_type>
                <<<device.blocks(input.size()), device.threads(), 0, stream>>>(
                    output, stat);
            core::checkCuda(cudaPeekAtLastError());

            return output;
        }

        /**
         * @brief Compute the backward pass of the Softmax function.
         *
         * @param error The gradient of the loss with respect to the output of
         * the Softmax layer.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The gradient with respect to the input.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            input_gradient.rebatchsize(error.batchsize(), stream);

            core::multiply<data_type>
                <<<device.blocks(error.size()), device.threads(), 0, stream>>>(
                    error, output, input_gradient);
            core::checkCuda(cudaPeekAtLastError());

            core::matrix_vector_multiplication<data_type>(
                device.cublas_handle(stream), 1.0, input_gradient,
                input_gradient.size() / size_last_dim, size_last_dim,
                CUBLAS_OP_N, one, 0.0, stat);
            core::checkCuda(cudaPeekAtLastError());

            softmax_backward_propagate<data_type>
                <<<device.blocks(input_gradient.size()), device.threads(), 0,
                   stream>>>(output, input_gradient, stat);
            core::checkCuda(cudaPeekAtLastError());

            return input_gradient;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            return output;
        }

        /**
         * @brief Get the gradient in respect to the input.
         */
        const core::Device_Tensor<data_type> &get_input_gradient()
            const override {
            return input_gradient;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += one.mem_size(true);
            size += output.mem_size();
            size += input_gradient.mem_size();

            // stat:
            size.fixed_size += stat.mem_size().fixed_size;
            size.variable_size +=
                output.sample_size() / size_last_dim * sizeof(data_type);

            return size;
        }
    };

}  // namespace activations
