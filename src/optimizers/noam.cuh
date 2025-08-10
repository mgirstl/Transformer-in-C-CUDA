/**
 * @file noam.cuh
 * @brief Implements the Noam learning rate scheduler.
 *
 * This file is part of the `optimizers` namespace, which implements
 * parameter optimizers
 */

#pragma once

#include <cmath>
#include <unordered_map>
#include <vector>

#include "../core.cuh"
#include "adam.cuh"
#include "none.cuh"
#include "optimizer.cuh"
#include "sgd.cuh"

/**
 * @internal
 *
 * @namespace optimizers
 * @brief Implements parameter optimizers.
 *
 * This specific file defines the Noam learning rate scheduler.
 *
 * @endinternal
 */
namespace optimizers {

    /**
     * @class Noam
     * @brief The Noam learning rate scheduler.
     *
     * The Noam learning rate scheduler adjusts the learning rate according to
     * the following formula:
     * @f[
     *     \alpha = \frac{\sigma}{\sqrt{d}} \cdot \min\left(
     *                  \frac{1}{\sqrt{t}}, \frac{t}{\tau \cdot \sqrt{\tau}}
     *              \right)
     * @f]
     * This allows for a warm-up phase where the learning rate increases
     * linearly, followed by a decay phase.
     *
     * Here:
     * - @f$ \alpha @f$ is the learning rate.
     * - @f$ \sigma @f$ is a scaling factor.
     * - @f$ d @f$ is the model dimension.
     * - @f$ t @f$ is the current step.
     * - @f$ \tau @f$ is the number of warm-up steps.
     *
     * @note The Noam scheduler is often used in conjunction with the Adam
     * optimizer.
     * @note In this concrete implementation, instead of setting the scaling
     * factor directly, the scaling factor is chosen so that a user-specified
     * initial learning rate is achieved in the first step.
     * @note The Noam learning rate scheduler is implemented as a wrapper so
     * that it can be used in conjunction with the implemented optimizers, e.g.,
     * Adam or SGD.
     */
    template <core::arithmetic_type data_type, template <typename> class otype>
    class Noam final : public Optimizer<data_type> {
        static_assert(
            optimizer_type<otype<data_type>, data_type>,
            "otype must satisfy the optimizers::optimizer_type concept!");

      private:
        core::index_type step;
        data_type model_dim;
        data_type warmup_steps;
        data_type learning_rate;
        data_type scale_factor;
        otype<data_type> optimizer;

      public:
        /**
         * @brief Construct a new Noam object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape of the weights.
         * @param kwargs Contains additional parameters:
         * - `model_dim` (no default, needs to be specified)
         * - `warmup_steps` (defaults to 4000)
         * - `learning_rate` (default, see below)
         * - Additional parameters for the used optimizer. See the documentation
         * of the specific optimizer for possible parameters and their default
         * values.
         * @param stream The CUDA stream used for launching the kernels.
         *
         * @note The `learning_rate` defaults to:
         * @f[
         *     \frac{1}{\sqrt{d}} \cdot \min\left(
         *         1, \frac{1}{\tau \cdot \sqrt{\tau}}
         *     \right)
         * @f]
         * where @f$ d @f$ is the model dimension and @f$ \tau @f$ is the number
         * of warm-up steps. This value corresponds to a scaling factor of 1 in
         * the original formula. This default is also used if the learning rate
         * is set to a negative value.
         */
        Noam(const core::Device &device,
             const std::vector<core::index_type> &shape,
             const std::unordered_map<std::string, data_type> &kwargs,
             const cudaStream_t stream = core::default_stream)
            : step{0},
              model_dim{kwargs.at("model_dim")},
              warmup_steps{kwargs.find("warmup_steps") != kwargs.end()
                               ? kwargs.at("warmup_steps") : 4000},
              learning_rate{kwargs.find("learning_rate") != kwargs.end()
                                && kwargs.at("learning_rate") > 0
                                ? kwargs.at("learning_rate")
                                : calculate_learning_rate(1)},
              scale_factor{learning_rate / calculate_learning_rate(1)},
              optimizer{
                  device,
                  shape,
                  [&kwargs](const data_type learning_rate) {
                      std::unordered_map<std::string, data_type> _kwargs = {
                          {"learning_rate", learning_rate}};
                      _kwargs.insert(kwargs.begin(), kwargs.end());
                      return _kwargs;
                  }(learning_rate),
                  stream} {}

        /**
         * @brief Computes the parameter update.
         *
         * @param weights The weights which will be updated.
         * @param gradient The gradient used for the update.
         * @param stream The CUDA stream used for launching the kernels.
         */
        void update(core::Device_Tensor<data_type> &weights,
                    const core::Device_Tensor<data_type> &gradient,
                    const cudaStream_t stream = core::default_stream) override {
            ++step;
            learning_rate = calculate_learning_rate();
            optimizer.set_learning_rate(learning_rate);
            optimizer.update(weights, gradient, stream);
        }

        /**
         * @brief Set the learning rate of the optimizer.
         *
         * @attention Setting the learning rate in Noam sets the scaling factor
         * so that in the first step, the learning rate corresponds to the given
         * value and not the current step!
         * @note If a negative learning rate is set, then a scaling factor of 1
         * is used instead.
         */
        void set_learning_rate(const data_type _learning_rate) override {
            if (_learning_rate > 0)
                scale_factor = _learning_rate / calculate_learning_rate(1);
            else
                scale_factor = 1;
            learning_rate = calculate_learning_rate();
            optimizer.set_learning_rate(learning_rate);
        }

        /**
         * @brief Get the learning_rate of the optimizer.
         */
        const data_type &get_learning_rate() const override {
            return learning_rate;
        }

        /**
         * @brief Get the optimizer used by Noam.
         */
        const otype<data_type> &get_optimizer() const { return optimizer; }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            return optimizer.mem_size();
        }

      private:
        /**
         * @brief Calculates the learning rate based on the current step.
         */
        data_type calculate_learning_rate() {
            return scale_factor / sqrt(model_dim) *
                   min(1 / sqrt(step),
                       step / (warmup_steps * sqrt(warmup_steps)));
        }

        /**
         * @brief Calculates the learning rate for a given step and scale
         * factor.
         *
         * @param _step The step for which to calculate the learning rate.
         * @param _scale_factor The scale factor to use in the calculation.
         * @return The calculated learning rate.
         */
        data_type calculate_learning_rate(const core::index_type _step,
                                          const data_type _scale_factor = 1.0) {
            return _scale_factor / sqrt(model_dim) *
                   min(1 / sqrt(_step),
                       _step / (warmup_steps * sqrt(warmup_steps)));
        }
    };

    /**
     * @brief Alias for Noam scheduler with Adam optimizer.
     *
     * @note This alias is necessary to easily use the Noam scheduler with the
     * Adam optimizer. Without this alias, the template instantiation would be
     * cumbersome and could lead to issues with fulfilling the Optimizer concept
     * due to template complexities.
     */
    template <core::arithmetic_type data_type>
    using Adam_with_Noam_Scheduler = Noam<data_type, Adam>;

    /**
     * @brief Alias for Noam scheduler with SGD optimizer.
     *
     * @note This alias is necessary to easily use the Noam scheduler with the
     * SGD optimizer. Without this alias, the template instantiation would be
     * cumbersome and could lead to issues with fulfilling the Optimizer concept
     * due to template complexities.
     */
    template <core::arithmetic_type data_type>
    using SGD_with_Noam_Scheduler = Noam<data_type, SGD>;

    /**
     * @brief Alias for Noam scheduler with no optimizer.
     *
     * @note This alias is necessary to easily use the Noam scheduler without
     * any optimizer. Without this alias, the template instantiation would be
     * cumbersome and could lead to issues with fulfilling the Optimizer concept
     * due to template complexities.
     */
    template <core::arithmetic_type data_type>
    using None_with_Noam_Scheduler = Noam<data_type, None>;

}  // namespace optimizers
