/**
 * @file relu.cuh
 * @brief Implements the ReLU function.
 *
 * This file is part of the `activations` namespace, which implements
 * differentiable activation functions.
 */

#pragma once

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
     * @brief This kernel calculates the forward pass of the ReLU function.
     *
     * This kernel applies the ReLU function element-wise to the input tensor.
     *
     * @param input The input tensor containing the values to be processed.
     * @param output The output tensor where the results will be stored in.
     * @param mask The mask tensor will store the mask indicating which elements
     * are greater than zero.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <core::arithmetic_type data_type>
    __global__ void relu_forward(const core::Kernel_Tensor<data_type> input,
                                 core::Kernel_Tensor<data_type> output,
                                 core::Kernel_Tensor<core::mask_type> mask) {
        core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        for (; idx < input.size(); idx += stride) {
            const data_type _input = input[idx];
            const core::mask_type _mask = _input > 0;

            if (_mask)
                output[idx] = _input;
            else
                output[idx] = 0;

            mask[idx] = _mask;
        }
    }

    /**
     * @brief This kernel calculates the backward pass of the ReLU function.
     *
     * This kernel applies the gradient of the ReLU function element-wise to the
     * error tensor.
     *
     * @param error The input tensor containing the error values from the next
     * layer.
     * @param input_gradient The output tensor where the gradient values will be
     * stored in.
     * @param mask A tensor that stores the mask indicating which elements were
     * greater than zero in the forward pass.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <core::arithmetic_type data_type>
    __global__ void relu_backward(
        const core::Kernel_Tensor<data_type> error,
        core::Kernel_Tensor<data_type> input_gradient,
        const core::Kernel_Tensor<core::mask_type> mask) {
        core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
        core::index_type stride = gridDim.x * blockDim.x;

        for (; idx < error.size(); idx += stride) {
            if (mask[idx])
                input_gradient[idx] = error[idx];
            else
                input_gradient[idx] = 0;
        }
    }

    /**
     * @class ReLU
     * @brief The ReLU function.
     *
     * The ReLU function is defined as:
     * @f[ f(x_j) = \max(x_j, 0) @f]
     * for each element @f$ x_j @f$ of the tensor @f$ x @f$.
     */
    template <core::arithmetic_type data_type>
    class ReLU final : public Activation<data_type> {
      private:
        const core::Device &device;
        core::Device_Tensor<core::mask_type> mask;
        core::Device_Tensor<data_type> output;
        core::Device_Tensor<data_type> input_gradient;

      public:
        /**
         * @brief Construct a new ReLU object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape (excluding the first axis) of the input/output
         * tensors.
         */
        ReLU(const core::Device &device,
             const std::vector<core::index_type> &shape,
             cudaStream_t /* unused */ = core::default_stream)
            : device{device},
              mask{core::NO_ALLOC, shape},
              output{core::NO_ALLOC, shape},
              input_gradient{core::NO_ALLOC, shape} {}

        /**
         * @brief Compute the forward pass of the ReLU function.
         *
         * @param input The input tensor containing the values to be processed.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            cudaStream_t stream = core::default_stream) override {
            output.rebatchsize(input.batchsize(), stream);
            mask.rebatchsize(input.batchsize(), stream);

            relu_forward<data_type>
                <<<device.blocks(input.size()), device.threads(), 0, stream>>>(
                    input, output, mask);
            core::checkCuda(cudaPeekAtLastError());

            return output;
        }

        /**
         * @brief Compute the backward pass of the ReLU function.
         *
         * @param error The gradient of the loss with respect to the output of
         * the ReLU layer.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The gradient with respect to the input.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            cudaStream_t stream = core::default_stream) override {
            input_gradient.rebatchsize(error.batchsize(), stream);

            relu_backward<data_type>
                <<<device.blocks(error.size()), device.threads(), 0, stream>>>(
                    error, input_gradient, mask);
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
            size += mask.mem_size();
            size += output.mem_size();
            size += input_gradient.mem_size();
            return size;
        }
    };

}  // namespace activations
