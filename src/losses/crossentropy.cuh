/**
 * @file crossentropy.cuh
 * @brief Implements the CrossEntropy function.
 *
 * This file is part of the `losses` namespace, which implements differentiable
 * loss functions.
 */

#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "../core.cuh"
#include "loss.cuh"

/**
 * @internal
 *
 * @namespace losses
 * @brief Implements differentiable loss functions.
 *
 * This specific file defines the CrossEntropy function.
 *
 * @endinternal
 */
namespace losses {

    /**
     * @brief This kernel calculates the forward and backward pass of the
     * CrossEntropy function.
     *
     * @note The forward and backward passes are combined for efficiency.
     *
     * @param input The input tensor containing the values to be processed. It
     * must have the shape @f$ [*, N] @f$.
     * @param output The output tensor where the loss of each input sample will
     * be stored in. It must have the shape @f$ [*] @f$.
     * @param input_gradient The tensor where the gradient of the loss with
     * respect to the input will be stored in. It must have the shape
     * @f$ [*, N] @f$.
     * @param target The expected target values. It must have the shape
     * @f$ [*] @f$.
     * @param ignore_index Specifies a target value that is ignored and does not
     * contribute to the `output` and `input_gradient`.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::float_type data_type, core::integer_type label_type>
    __global__ void crossentropy_forward_backward_combined(
        const core::Kernel_Tensor<data_type> input,
        core::Kernel_Tensor<data_type> output,
        core::Kernel_Tensor<data_type> input_gradient,
        const core::Kernel_Tensor<label_type> target,
        const label_type ignore_index) {
        core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
        core::index_type stride = gridDim.x * blockDim.x;

        const core::index_type size_last_dim = input.shape(input.rank() - 1);
        const core::index_type batchsize = input.size() / size_last_dim;
        const data_type factor = data_type(1.) / batchsize;

        constexpr data_type epsilon =
            std::numeric_limits<data_type>::epsilon() *
            std::numeric_limits<data_type>::epsilon();

        for (; idx < batchsize; idx += stride) {
            const label_type _target = target[idx];
            if (_target != ignore_index) {
                const core::index_type _idx = idx * size_last_dim + _target;
                const data_type _input = input[_idx] + epsilon;
                output[idx] = -std::log(_input);
                input_gradient[_idx] = -factor / _input;
            } else
                output[idx] = 0;
        }
    }

    /**
     * @class CrossEntropy
     * @brief The CrossEntropy function.
     *
     * The CrossEntropy function is defined as:
     * @f[ f(\hat{y}) = - \sum_j y_j \log \hat{y}_j @f]
     * for each sample. Here @f$ y @f$ is the one-hot encoded target vector and
     * @f$ \hat{y} @f$ the prediction of a network.
     *
     * @note @f$ \hat{y} @f$ needs to sum up to 1 (be a valid probability
     * distribution).
     * @note The CrossEntropy implemented here always assumes that the last
     * input dimension is the dimension where the sum is applied.
     * @note The CrossEntropy returns the loss of each sample. If the reduced
     * loss, i.e., the average loss, is needed, an extra reduction step is
     * required.
     * @note The CrossEntropy implemented here does not correspond to the
     * `torch.nn.CrossEntropyLoss` in PyTorch but to
     * `torch.nn.NLLLoss(torch.log(...))`.
     */
    template <core::float_type data_type, core::integer_type label_type>
    class CrossEntropy final : public Loss<data_type, label_type> {
      private:
        const core::Device &device;
        core::index_type ignore_index;
        core::index_type size_last_dim;
        core::Device_Tensor<data_type> output;
        core::Device_Tensor<data_type> input_gradient;

      public:
        /**
         * @brief Construct a new CrossEntropy object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape (excluding the first axis) of the input/output
         * tensors.
         * @param ignore_index Specifies a target value that is ignored and does
         * not contribute to the `output` and `input_gradient`.
         */
        CrossEntropy(const core::Device &device,
                     const std::vector<core::index_type> &shape,
                     const label_type ignore_index =
                         std::numeric_limits<label_type>::max())
            : device{device},
              ignore_index{ignore_index},
              size_last_dim{shape.size() == 0 ? 1 : shape[shape.size() - 1]},
              output{core::NO_ALLOC, std::vector<core::index_type>(
                                         shape.begin(), shape.end() - 1)},
              input_gradient{core::NO_ALLOC, shape} {}

        /**
         * @brief Compute the forward pass of the CrossEntropy function.
         *
         * @param input The input tensor containing the values to be processed.
         * @param target The target tensor containing the targets of the current
         * batch.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const core::Device_Tensor<label_type> &target,
            const cudaStream_t stream = core::default_stream) override {
            output.rebatchsize(input.batchsize(), stream);
            input_gradient.rebatchsize(input.batchsize(), stream);

            core::checkCuda(cudaMemsetAsync(
                input_gradient.data(), 0,
                input_gradient.size() * sizeof(data_type), stream));

            crossentropy_forward_backward_combined<data_type, label_type>
                <<<device.blocks(input.size() / size_last_dim),
                   device.threads(), 0, stream>>>(input, output, input_gradient,
                                                  target, ignore_index);
            core::checkCuda(cudaPeekAtLastError());

            return output;
        }

        /**
         * @brief Compute the backward pass of the CrossEntropy function.
         *
         * @return The gradient with respect to the input.
         */
        const core::Device_Tensor<data_type> &backward(
            const cudaStream_t /* unused */ = core::default_stream) override {
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
            size += output.mem_size();
            size += input_gradient.mem_size();
            return size;
        }
    };

}  // namespace losses
