/**
 * @file adam.cuh
 * @brief Implements Adam (Adaptive Moment Estimation).
 *
 * This file is part of the `optimizers` namespace, which implements
 * parameter optimizers
 */

#pragma once

#include <cmath>
#include <limits>
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
 * This specific file defines the Adam (Adaptive Moment Estimation) optimizer.
 *
 * @endinternal
 */
namespace optimizers {

    /**
     * @brief This kernel calculates the Adam update.
     *
     * @param weights The tensor which will be updated.
     * @param gradient The gradient values to add to the weights.
     * @param alpha The learning rate used for scaling the gradient.
     * @param first_moment The first moment calculated over the previous
     * updates.
     * @param second_moment The second moment calculated over the previous
     * updates.
     * @param mu The scaling parameter for the first moment.
     * @param rho The scaling parameter for the second moment.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <core::float_type data_type>
    __global__ void adam_update(core::Kernel_Tensor<data_type> weights,
                                const core::Kernel_Tensor<data_type> gradient,
                                const data_type alpha,
                                core::Kernel_Tensor<data_type> first_moment,
                                core::Kernel_Tensor<data_type> second_moment,
                                const data_type mu, const data_type rho) {
        const core::index_type thread_id =
            blockDim.x * blockIdx.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        constexpr data_type epsilon =
            std::numeric_limits<data_type>::epsilon() *
            std::numeric_limits<data_type>::epsilon();

        using vector_type = typename core::vector_type<data_type>::type;
        constexpr int elements_per_vector =
            core::vector_type<data_type>::elements_per_vector;

        const core::index_type vmax = weights.size() / elements_per_vector;

        for (core::index_type idx = thread_id; idx < vmax; idx += stride) {
            vector_type *weights_vec =
                reinterpret_cast<vector_type *>(weights.data());
            vector_type *first_moment_vec =
                reinterpret_cast<vector_type *>(first_moment.data());
            vector_type *second_moment_vec =
                reinterpret_cast<vector_type *>(second_moment.data());
            const vector_type *gradient_vec =
                reinterpret_cast<const vector_type *>(gradient.data());

            vector_type w = weights_vec[idx];
            vector_type v = first_moment_vec[idx];
            vector_type r = second_moment_vec[idx];
            const vector_type g = gradient_vec[idx];

            v.x = mu * v.x + (1 - mu) * g.x;
            r.x = rho * r.x + (1 - rho) * g.x * g.x;
            w.x -= alpha * v.x / (sqrt(r.x) + epsilon);

            v.y = mu * v.y + (1 - mu) * g.y;
            r.y = rho * r.y + (1 - rho) * g.y * g.y;
            w.y -= alpha * v.y / (sqrt(r.y) + epsilon);

            if constexpr (elements_per_vector > 2) {
                v.z = mu * v.z + (1 - mu) * g.z;
                r.z = rho * r.z + (1 - rho) * g.z * g.z;
                w.z -= alpha * v.z / (sqrt(r.z) + epsilon);

                v.w = mu * v.w + (1 - mu) * g.w;
                r.w = rho * r.w + (1 - rho) * g.w * g.w;
                w.w -= alpha * v.w / (sqrt(r.w) + epsilon);
            }

            weights_vec[idx] = w;
            first_moment_vec[idx] = v;
            second_moment_vec[idx] = r;
        }

        for (core::index_type idx = thread_id + vmax * elements_per_vector;
             idx < weights.size(); idx += stride) {
            data_type v = first_moment[idx];
            data_type r = second_moment[idx];
            const data_type g = gradient[idx];

            v = mu * v + (1 - mu) * g;
            r = rho * r + (1 - rho) * g * g;

            weights[idx] -= alpha * v / (sqrt(r) + epsilon);
            first_moment[idx] = v;
            second_moment[idx] = r;
        }
    }

    /**
     * @class Adam
     * @brief The Adam (Adaptive Moment Estimation) optimizer.
     *
     * The Adam optimizer applies the following update rules:
     * @f[ m_t = \mu m_{t-1} + (1 - \mu) \nabla_\theta L_t @f]
     * @f[ v_t = \rho v_{t-1} + (1 - \rho) (\nabla_\theta L_t)^2 @f]
     * @f[ \hat{m}_t = \frac{m_t}{1 - \mu^t} @f]
     * @f[ \hat{v}_t = \frac{v_t}{1 - \rho^t} @f]
     * @f[
     *     \theta \leftarrow
     *         \theta - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
     * @f]
     * for each element of the parameters @f$ \theta @f$. Here, @f$ \alpha @f$
     * is the learning rate, @f$ \nabla_\theta L @f$ is the gradient of the loss
     * @f$ L @f$ with respect to the parameters @f$ \theta @f$, @f$ \mu @f$ and
     * @f$ \rho @f$ are the exponential decay rates for the moment estimates,
     * @f$ m_t @f$ and @f$ v_t @f$ are the first and second moment estimates,
     * and @f$ \epsilon @f$ is a small constant for numerical stability.
     *
     * @note @f$ m_0 @f$ and @f$ v_0 @f$ are initialized to 0 for @f$ t=0 @f$.
     */
    template <core::float_type data_type>
    class Adam final : public Optimizer<data_type> {
      private:
        const core::Device &device;
        data_type learning_rate;
        data_type mu;
        data_type rho;
        core::index_type step;
        core::Device_Tensor<data_type> first_moment;
        core::Device_Tensor<data_type> second_moment;

      public:
        /**
         * @brief Construct a new Adam object
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape of the weights.
         * @param kwargs Contains additional parameters:
         * - `learning_rate` (defaults to 1e-3)
         * - `mu` (defaults to 0.9)
         * - `rho` (defaults to 0.999).
         * @param stream The CUDA stream used for launching the kernels.
         */
        Adam(const core::Device &device,
             const std::vector<core::index_type> &shape,
             const std::unordered_map<std::string, data_type> &kwargs,
             const cudaStream_t stream = core::default_stream)
            : device{device},
              learning_rate{kwargs.find("learning_rate") != kwargs.end()
                                ? kwargs.at("learning_rate") : 1e-3},
              mu{kwargs.find("mu") != kwargs.end() ? kwargs.at("mu") : 0.9},
              rho{kwargs.find("rho") != kwargs.end()
                      ? kwargs.at("rho") : 0.999},
              step{0},
              first_moment{shape},
              second_moment{shape} {
            core::checkCuda(cudaMemsetAsync(
                first_moment.data(), 0, first_moment.size() * sizeof(data_type),
                stream));
            core::checkCuda(cudaPeekAtLastError());

            core::checkCuda(cudaMemsetAsync(
                second_moment.data(), 0,
                second_moment.size() * sizeof(data_type), stream));
            core::checkCuda(cudaPeekAtLastError());
        }

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

            ++step;
            data_type alpha =
                learning_rate * sqrt(1 - pow(rho, step)) / (1 - pow(mu, step));

            adam_update<data_type>
                <<<device.blocks(_size), device.threads(), 0, stream>>>(
                    weights, gradient, alpha, first_moment, second_moment, mu,
                    rho);
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

        /**
         * @brief Get the first_moment of the optimizer.
         */
        const core::Device_Tensor<data_type> &get_first_moment() const {
            return first_moment;
        }

        /**
         * @brief Get the second_moment of the optimizer.
         */
        const core::Device_Tensor<data_type> &get_second_moment() const {
            return second_moment;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += first_moment.mem_size(true);
            size += second_moment.mem_size(true);
            return size;
        }
    };

}  // namespace optimizers
