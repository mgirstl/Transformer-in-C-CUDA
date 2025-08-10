/**
 * @file loss.cuh
 * @brief Implements the interface for differentiable loss functions.
 *
 * This file is part of the `losses` namespace, which implements differentiable
 * loss functions.
 */

#pragma once

#include <type_traits>

#include "../core.cuh"

/**
 * @namespace losses
 * @brief Implements differentiable loss functions.
 *
 * @internal
 * This specific file defines the Loss interface.
 * @endinternal
 */
namespace losses {

    /**
     * @interface Loss
     * @brief Interface for differentiable loss functions.
     */
    template <core::arithmetic_type data_type, core::arithmetic_type label_type>
    class Loss {
      public:
        /**
         * @brief Compute the forward pass of the function.
         */
        virtual const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const core::Device_Tensor<label_type> &target,
            const cudaStream_t stream) = 0;

        /**
         * @brief Compute the backwards pass of the function.
         */
        virtual const core::Device_Tensor<data_type> &backward(
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
         * @brief Destroy the Loss object
         *
         * Ensures correct behavior when deleting derived classes.
         */
        virtual ~Loss() {}
    };

}  // namespace losses
