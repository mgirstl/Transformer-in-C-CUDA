/**
 * @file identity.cuh
 * @brief Implements the identity function.
 *
 * This file is part of the `activations` namespace, which implements
 * differentiable activation functions.
 */

#pragma once

#include "../core.cuh"
#include "activation.cuh"

/**
 * @internal
 *
 * @namespace activations
 * @brief Implements differentiable activation functions.
 *
 * This specific file defines the identity function.
 *
 * @endinternal
 */
namespace activations {

    /**
     * @class Identity
     * @brief The Identity function.
     */
    template <core::arithmetic_type data_type>
    class Identity final : public Activation<data_type> {
      private:
        const core::Device_Tensor<data_type> *output;
        const core::Device_Tensor<data_type> *input_gradient;

      public:
        Identity() = default;

        /**
         * @brief Construct a new Identity object.
         *
         * @note This constructor only exists so that the Identity function can
         * be used as drop in replacement for the other activation functions.
         */
        Identity(const core::Device & /* unused */,
                 const std::vector<core::index_type> & /* unused */,
                 const cudaStream_t /* unused */ = core::default_stream) {}

        /**
         * @brief Compute the forward pass of the identity function.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const cudaStream_t /* unused */ = core::default_stream) override {
            output = &input;
            return input;
        }

        /**
         * @brief Compute the backward pass of the identity function.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t /* unused */ = core::default_stream) override {
            input_gradient = &error;
            return error;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            return *output;
        }

        /**
         * @brief Get the gradient in respect to the input.
         */
        const core::Device_Tensor<data_type> &get_input_gradient()
            const override {
            return *input_gradient;
        }
    };

}  // namespace activations
