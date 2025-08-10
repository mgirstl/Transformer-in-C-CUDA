/**
 * @file activation.cuh
 * @brief Implements the interface for differentiable activation functions.
 *
 * This file is part of the `activations` namespace, which implements
 * differentiable activation functions.
 */

#pragma once

#include "../core.cuh"

/**
 * @namespace activations
 * @brief Implements differentiable activation functions.
 *
 * @internal
 * This specific file defines the Activation interface.
 * @endinternal
 */
namespace activations {

    /**
     * @interface Activation
     * @brief Interface for differentiable activation functions.
     */
    template <core::arithmetic_type data_type>
    class Activation {
      public:
        /**
         * @brief Compute the forward pass of the function.
         */
        virtual const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const cudaStream_t stream) = 0;

        /**
         * @brief Compute the backward pass of the function.
         */
        virtual const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t stream) = 0;

        /**
         * @brief Get the output of the forward pass.
         */
        virtual const core::Device_Tensor<data_type> &get_output() const = 0;

        /**
         * @brief Get the gradient in respect to the input.
         */
        virtual const core::Device_Tensor<data_type> &get_input_gradient()
            const = 0;

        /**
         * @brief Returns the memory footprint of the object on GPU.
         *
         * @note This function should be overridden by derived classes if
         * the derived class allocates memory on the GPU.
         */
        virtual const core::Mem_Size mem_size(
            const bool /* unused */ = false) const {
            return {0, 0};
        }

        /**
         * @brief Destroy the Activation object.
         *
         * Ensures correct behavior when deleting derived classes.
         */
        virtual ~Activation() {}
    };

}  // namespace activations
