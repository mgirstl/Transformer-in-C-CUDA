/**
 * @file reshape.cuh
 * @brief Implements the Reshape class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../core.cuh"
#include "../optimizers/none.cuh"
#include "layer.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the Reshape class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @class Reshape.
     * @brief The Reshape layer.
     *
     * The Reshape layer will reshape a tensor with shape @f$ [N, *] @f$ to
     * @f$ [N, *'] @f$.
     * The @f$ * @f$ symbol denotes any number of dimensions. The only condition
     * is, that the number of elements in both tensor is the same.
     *
     * @tparam data_type The data type of the tensor elements.
     */
    template <core::arithmetic_type data_type>
    class Reshape final : public Layer<data_type, data_type, optimizers::None> {
      private:
        std::vector<core::index_type> input_shape;
        std::vector<core::index_type> output_shape;
        core::Device_Tensor<data_type> output;
        core::Device_Tensor<data_type> input_gradient;

      public:
        /**
         * @brief Construct a new Reshape object
         *
         * @param input_shape The shape (excluding the first axis) of the input
         * tensors.
         * @param output_shape The shape (excluding the first axis) of the
         * output tensors.
         *
         * @throws std::invalid_argument if the input sample size and the
         * output sample size do not match.
         */
        Reshape(const std::vector<core::index_type> &input_shape,
                const std::vector<core::index_type> &output_shape)
            : input_shape(input_shape.size() + 1),
              output_shape(output_shape.size() + 1) {
            std::copy(input_shape.begin(), input_shape.end(),
                      this->input_shape.begin() + 1);
            std::copy(output_shape.begin(), output_shape.end(),
                      this->output_shape.begin() + 1);

            core::index_type input_sample_size = std::accumulate(
                this->input_shape.begin() + 1, this->input_shape.end(),
                core::index_type(1), std::multiplies<core::index_type>());
            core::index_type output_sample_size = std::accumulate(
                this->output_shape.begin() + 1, this->output_shape.end(),
                core::index_type(1), std::multiplies<core::index_type>());

            if (input_sample_size != output_sample_size)
                throw std::invalid_argument(
                    "The input sample size and the output sample size do not "
                    "match!");
        }

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream used for copying.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const cudaStream_t stream = core::default_stream) override {
            output.copy_from(input, stream);
            output_shape[0] = input.batchsize();
            output.reshape(output_shape);
            return output;
        }

        /**
         * @brief Compute the backward pass of the layer.
         *
         * @param error The error tensor from the next layer.
         * @param stream The CUDA stream used for copying.
         * @return The gradient with respect to the input.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            input_gradient.copy_from(error, stream);
            input_shape[0] = error.batchsize();
            input_gradient.reshape(input_shape);
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

}  // namespace layers
