/**
 * @file sgd.cuh
 * @brief Implements SGD (Stochastic Gradient Descent).
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
 * This specific file defines the SGD (Stochastic Gradient Descent) optimizer.
 *
 * @endinternal
 */
namespace optimizers {

    /**
     * @brief This kernel calculates the SGD update.
     *
     * @param weights The tensor which will be updated.
     * @param gradient The gradient values to add to the weights.
     * @param learning_rate The learning rate used for scaling the gradient.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <core::float_type data_type>
    __global__ void sgd_update(core::Kernel_Tensor<data_type> weights,
                               const core::Kernel_Tensor<data_type> gradient,
                               const data_type learning_rate) {
        const core::index_type thread_id =
            blockDim.x * blockIdx.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        using vector_type = typename core::vector_type<data_type>::type;
        constexpr int elements_per_vector =
            core::vector_type<data_type>::elements_per_vector;

        const core::index_type vmax = weights.size() / elements_per_vector;

        for (core::index_type idx = thread_id; idx < vmax; idx += stride) {
            vector_type *weights_vec =
                reinterpret_cast<vector_type *>(weights.data());
            const vector_type *gradient_vec =
                reinterpret_cast<const vector_type *>(gradient.data());

            vector_type w = weights_vec[idx];
            const vector_type g = gradient_vec[idx];

            w.x -= learning_rate * g.x;
            w.y -= learning_rate * g.y;
            if constexpr (elements_per_vector > 2) {
                w.z -= learning_rate * g.z;
                w.w -= learning_rate * g.w;
            }

            weights_vec[idx] = w;
        }

        for (core::index_type idx = thread_id + vmax * elements_per_vector;
             idx < weights.size(); idx += stride) {
            weights[idx] -= learning_rate * gradient[idx];
        }
    }

    /**
     * @class SGD
     * @brief The SGD (Stochastic Gradient Descent) optimizer.
     *
     * The SGD (Stochastic Gradient Descent) optimizer applies the following
     * update rule:
     * @f[ \theta \leftarrow \theta - \alpha \nabla_\theta L @f]
     * for each element of the parameters @f$ \theta @f$. Here, @f$ \alpha @f$
     * is the learning rate and @f$ \nabla_\theta L @f$ is the gradient of the
     * loss @f$ L @f$ with respect to the parameters @f$ \theta @f$.
     */
    template <core::float_type data_type>
    class SGD final : public Optimizer<data_type> {
      private:
        const core::Device &device;
        data_type learning_rate;

      public:
        /**
         * @brief Construct a new SGD object.
         *
         * @note The unused parameters exists so that SGD can be a drop in
         * replacement for the other optimizers.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape of the weights. (Here unused.)
         * @param kwargs Contains additional parameters:
         * - `learning_rate` (defaults to 1e-3).
         */
        SGD(const core::Device &device,
            const std::vector<core::index_type> & /* unused */,
            const std::unordered_map<std::string, data_type> &kwargs,
            const cudaStream_t /* unused */ = core::default_stream)
            : device{device},
              learning_rate{kwargs.find("learning_rate") != kwargs.end()
                                ? kwargs.at("learning_rate")
                                : 1e-3} {}

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
            constexpr int elements_per_vector =
                core::vector_type<data_type>::elements_per_vector;
            core::index_type _size =
                (weights.size() + elements_per_vector - 1) /
                elements_per_vector;

            sgd_update<data_type>
                <<<device.blocks(_size), device.threads(), 0, stream>>>(
                    weights, gradient, learning_rate);
            core::checkCuda(cudaPeekAtLastError());
        }

        /**
         * @brief Set the learning_rate of the optimizer.
         */
        void set_learning_rate(const data_type _learning_rate) override {
            learning_rate = _learning_rate;
        }

        /**
         * @brief Get the learning_rate of the optimizer.
         */
        const data_type &get_learning_rate() const override {
            return learning_rate;
        }
    };

}  // namespace optimizers
