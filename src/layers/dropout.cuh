/**
 * @file dropout.cuh
 * @brief Implements the Dropout class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <type_traits>
#include <vector>

#include "../core.cuh"
#include "../optimizers/none.cuh"
#include "../utils/random_number_generator.cuh"
#include "layer.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the Dropout class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @brief This kernel calculates the forward pass of the Dropout layer.
     *
     * This kernel sets elements of the input tensor to zero with a certain
     * probability. Additionally, a scaling is applied so that the intensity of
     * the layer (the expected sum of the propagated values) remains consistent
     * regardless of the dropout probability.
     *
     * @param input The input tensor containing the values to be processed.
     * @param output The output tensor where the results will be stored in.
     * @param mask The mask tensor will store the mask indicating which elements
     * were not set to zero.
     * @param probability The probability of setting an element to zero.
     * @param rng_state Pointer to the array of curand states. The array must be
     * at least as large as the maximum number of unique threads that will be
     * launched.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <core::float_type data_type>
    __global__ void dropout_forward(const core::Kernel_Tensor<data_type> input,
                                    core::Kernel_Tensor<data_type> output,
                                    core::Kernel_Tensor<data_type> mask,
                                    const data_type probability,
                                    curandStatePhilox4_32_10_t *rng_state) {
        const core::index_type thread_id =
            blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        using vector_type = typename core::vector_type<data_type>::type;
        constexpr int elements_per_vector =
            core::vector_type<data_type>::elements_per_vector;

        const core::index_type vmax = input.size() / elements_per_vector;

        curandStatePhilox4_32_10_t _rng_state = rng_state[thread_id];
        vector_type *mask_vec = reinterpret_cast<vector_type *>(mask.data());
        vector_type *output_vec =
            reinterpret_cast<vector_type *>(output.data());
        const vector_type *input_vec =
            reinterpret_cast<const vector_type *>(input.data());

        const data_type inv_probability = 1 / (1 - probability);

        for (core::index_type idx = thread_id; idx < vmax; idx += stride) {
            const vector_type val = core::uniform<vector_type>(_rng_state);
            const vector_type i = input_vec[idx];
            vector_type o = output_vec[idx];
            vector_type m;
            m.x = val.x > probability;
            m.y = val.y > probability;
            o.x = m.x * i.x * inv_probability;
            o.y = m.y * i.y * inv_probability;

            if constexpr (elements_per_vector > 2) {
                m.z = val.z > probability;
                m.w = val.w > probability;
                o.z = m.z * i.z * inv_probability;
                o.w = m.w * i.w * inv_probability;
            }

            mask_vec[idx] = m;
            output_vec[idx] = o;
        }

        for (core::index_type idx = thread_id + vmax * elements_per_vector;
             idx < input.size(); idx += stride) {
            const data_type val = core::uniform<data_type>(_rng_state);
            const data_type m = val > probability;
            mask[idx] = m;
            output[idx] = m * input[idx] / (1 - probability);
        }

        rng_state[thread_id] = _rng_state;
    }

    /**
     * @brief This kernel calculates the backward pass of the Dropout layer.
     *
     * @param error The input tensor containing the error values from the next
     * layer.
     * @param input_gradient The output tensor where the gradient values will be
     * stored in.
     * @param mask A tensor that stores the mask indicating which elements were
     * not set to zero in the forward pass.
     * @param probability The probability with which the elements were set to
     * zero in the forward pass.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <core::arithmetic_type data_type>
    __global__ void dropout_backward(
        const core::Kernel_Tensor<data_type> error,
        core::Kernel_Tensor<data_type> input_gradient,
        const core::Kernel_Tensor<data_type> mask,
        const data_type probability) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        for (; idx < error.size(); idx += stride)
            input_gradient[idx] = mask[idx] * error[idx] / (1 - probability);
    }

    /**
     * @class Dropout
     * @brief The Dropout layer.
     *
     * The Dropout layer randomly sets elements of the input tensor to zero.
     * This helps to stabilize the training and decouple neurons during the
     * training process.
     *
     * @note This layer is in training mode after initialization. This means
     * in this particular case that in the forward pass dropout is applied. In
     * contrast, in evaluation mode the layer behaves like the identity
     * function.
     *
     * @tparam data_type The data type of the tensor elements.
     *
     * @note For convenience, the elements of the mask tensor have the type
     * `data_type` instead of `core::mask_type`. The reason for this is that one
     * can use the same dropout_forward function for all floating-point types.
     */
    template <core::float_type data_type>
    class Dropout final : public Layer<data_type, data_type, optimizers::None> {
      private:
        core::Device &device;
        data_type probability;
        utils::Random_Number_Generator<curandStatePhilox4_32_10_t> &rng;
        core::Device_Tensor<data_type> mask;
        core::Device_Tensor<data_type> output;
        core::Device_Tensor<data_type> input_gradient;
        const core::Device_Tensor<data_type> *eval_output;
        const core::Device_Tensor<data_type> *eval_input_gradient;
        bool training;

      public:
        /**
         * @brief Construct a new Dropout object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape (excluding the first axis) of the input/output
         * tensors.
         * @param probability The probability of an element being set to zero.
         * @param stream The CUDA stream used for initializing the random number
         * generator.
         */
        Dropout(core::Device &device,
                const std::vector<core::index_type> &shape,
                const data_type probability = 0.5,
                const cudaStream_t stream = core::default_stream)
            : device{device},
              probability{probability},
              rng{utils::Random_Number_Generator<
                  curandStatePhilox4_32_10_t>::get_instance(device, stream)},
              mask{core::NO_ALLOC, shape},
              output{core::NO_ALLOC, shape},
              input_gradient{core::NO_ALLOC, shape},
              training{true} {}

        /**
         * @brief Computes the forward pass of the layer.
         *
         * @note This function will only apply Dropout in training mode. In
         * evaluation mode, this function is equivalent to the identity.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const cudaStream_t stream = core::default_stream) override {
            if (!training) {
                eval_output = &input;
                return input;
            }

            output.rebatchsize(input.batchsize(), stream);
            mask.rebatchsize(input.batchsize(), stream);

            constexpr int elements_per_vector =
                core::vector_type<data_type>::elements_per_vector;
            core::index_type _blocks =
                (input.size() + elements_per_vector - 1) / elements_per_vector;
            dropout_forward<data_type>
                <<<device.blocks(_blocks), device.threads(), 0, stream>>>(
                    input, output, mask, probability, rng.states());
            core::checkCuda(cudaPeekAtLastError());

            return output;
        }

        /**
         * @brief Computes the backward pass of the layer.
         *
         * @param error The error tensor from the next layer.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The gradient with respect to the input.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            if (!training) {
                eval_input_gradient = &error;
                return error;
            }

            input_gradient.rebatchsize(error.batchsize(), stream);

            dropout_backward<data_type>
                <<<device.blocks(error.size()), device.threads(), 0, stream>>>(
                    error, input_gradient, mask, probability);
            core::checkCuda(cudaPeekAtLastError());

            return input_gradient;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            if (training)
                return output;
            else
                return *eval_output;
        }

        /**
         * @brief Get the gradient with respect to the input.
         */
        const core::Device_Tensor<data_type> &get_input_gradient()
            const override {
            if (training)
                return input_gradient;
            else
                return *eval_input_gradient;
        }

        /**
         * @brief Get the mask used in the last forward pass.
         */
        const core::Device_Tensor<data_type> &get_mask() const { return mask; }

        /**
         * @brief Set the mask used in the next backward pass.
         *
         * @note This mask will be overwritten in the next forward pass.
         */
        void set_mask(const core::Device_Tensor<data_type> &_mask) {
            mask = _mask;
        }

        /**
         * @brief Sets the layer into training mode.
         *
         * In training mode, the dropout with the given probability will be
         * applied.
         *
         * This layer is initially set to training mode.
         */
        void train() override { training = true; }

        /**
         * @brief Sets the layer into evaluation mode.
         *
         * In evaluation mode, the layer behaves like the identity function.
         *
         * This layer is initially set to training mode.
         */
        void eval() override { training = false; }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += rng.mem_size(true);
            size += mask.mem_size();
            size += output.mem_size();
            size += input_gradient.mem_size();
            return size;
        }
    };

}  // namespace layers
