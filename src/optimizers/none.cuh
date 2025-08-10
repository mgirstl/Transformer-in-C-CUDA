/**
 * @file none.cuh
 * @brief Implements the None function.
 *
 * This file is part of the `optimizers` namespace, which implements
 * parameter optimizers
 */

#pragma once

#include <unordered_map>
#include <vector>

#include "../core.cuh"
#include "optimizer.cuh"

/**
 * @internal
 *
 * @namespace optimizers
 * @brief Implements parameter optimizers.
 *
 * This specific file defines the None optimizer.
 *
 * @endinternal
 */
namespace optimizers {

    /**
     * @class None
     * @brief An optimizer which does not change the parameters.
     */
    template <core::arithmetic_type data_type>
    class None final : public Optimizer<data_type> {
      private:
        data_type learning_rate = 0;

      public:
        None() = default;

        /**
         * @brief Construct a new None object.
         *
         * @note This constructor only exists so that the None optimizer can
         * be used as drop in replacement for the other optimizers.
         */
        None(const core::Device & /* unused */,
             const std::vector<core::index_type> & /* unused */,
             const std::unordered_map<std::string, data_type> & /* unused */,
             const cudaStream_t /* unused */ = core::default_stream) {}

        /**
         * @brief Computes the parameter update.
         */
        void update(core::Device_Tensor<data_type> & /* unused */,
                    const core::Device_Tensor<data_type> & /* unused */,
                    const cudaStream_t stream = core::default_stream) override {
        }

        /**
         * @brief Set the learning_rate of the optimizer.
         */
        void set_learning_rate(const data_type /* unused */) override {}

        /**
         * @brief Get the learning_rate of the optimizer.
         */
        const data_type &get_learning_rate() const override {
            return learning_rate;
        }
    };

}  // namespace optimizers
