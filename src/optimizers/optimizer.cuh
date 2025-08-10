/**
 * @file optimizer.cuh
 * @brief Implements the interface for parameter optimizers.
 *
 * This file is part of the `optimizers` namespace, which implements
 * parameter optimizers
 */

#pragma once

#include <type_traits>

#include "../core.cuh"

/**
 * @namespace optimizers
 * @brief Implements parameter optimizers.
 *
 * @internal
 * This specific file defines the Optimizer interface.
 * @endinternal
 */
namespace optimizers {

    /**
     * @interface Optimizer
     * @brief Interface for parameter optimizers.
     */
    template <core::arithmetic_type data_type>
    class Optimizer {
      public:
        /**
         * @brief Computes the parameter update.
         */
        virtual void update(core::Device_Tensor<data_type> &weights,
                            const core::Device_Tensor<data_type> &gradient,
                            const cudaStream_t stream) = 0;

        /**
         * @brief Set the learning_rate of the optimizer.
         */
        virtual void set_learning_rate(const data_type learning_rate) = 0;

        /**
         * @brief Get the learning_rate of the optimizer.
         */
        virtual const data_type &get_learning_rate() const = 0;

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
         * @brief Destroy the Optimizer object
         *
         * Ensures correct behavior when deleting derived classes.
         */
        virtual ~Optimizer() {}
    };

    /**
     * @concept optimizer_type
     * @brief Concept that checks if a type is derived from the optimizer
     * interface.
     */
    template <typename T, typename data_type>
    concept optimizer_type = std::is_base_of_v<Optimizer<data_type>, T>;

}  // namespace optimizers
