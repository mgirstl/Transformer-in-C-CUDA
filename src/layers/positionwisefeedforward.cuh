/**
 * @file positionwisefeedforward.cuh
 * @brief Implements the PositionwiseFeedForward class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "../activations/relu.cuh"
#include "../core.cuh"
#include "dense.cuh"
#include "dropout.cuh"
#include "layer.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the PositionwiseFeedForward class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @class PositionwiseFeedForward
     * @brief The Positionwise Feed Forward layer.
     *
     * The Positionwise Feed Forward layer applies a sequence of transformations
     * to the input tensor:
     * 1. A Dense layer.
     * 2. A ReLU activation.
     * 3. A Dropout layer.
     * 4. Another Dense layer.
     *
     * @note This layer is in training mode after initialization. This means
     * in this particular case that in the forward pass dropout is applied, and
     * in the backward pass the trainable parameters will be updated. In
     * contrast, in evaluation mode the dropout behaves like the identity
     * function and the parameter update is omitted.
     *
     * @tparam data_type The data type of the tensor elements.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::float_type data_type,
              template <typename> class optimizer_type>
    class PositionwiseFeedForward final
        : public Layer<data_type, data_type, optimizer_type> {
      private:
        Dense<data_type, optimizer_type> dense_1;
        activations::ReLU<data_type> relu;
        Dropout<data_type> dropout;
        Dense<data_type, optimizer_type> dense_2;

      public:
        /**
         * @brief Construct a new PositionwiseFeedForward object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape (excluding the first axis) of the input/output
         * tensors, i.e., @f$ [*, M] @f$.
         * @param hidden_shape The shape (excluding the first axis) of the
         * features in the hidden dimension of the Positionwise Feed Forward
         * layer, i.e., @f$ [*, N] @f$.
         * @param optimizer_kw Will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param dropout The probability that an element in the dropout layer
         * is set to zero.
         * @param stream The CUDA stream used for launching the kernels.
         *
         * @note The @f$ * @f$ symbol denotes any number of values but they need
         * to be the same in all shapes where it is used.
         */
        PositionwiseFeedForward(
            core::Device &device,
            const std::vector<core::index_type> &input_shape,
            const std::vector<core::index_type> &hidden_shape,
            const std::unordered_map<std::string, data_type> &optimizer_kw = {},
            const data_type dropout = 0.1,
            const cudaStream_t stream = core::default_stream)
            : dense_1{device,
                      input_shape,
                      hidden_shape,
                      optimizer_kw,
                      true,
                      stream},
              relu{device,
                   hidden_shape},
              dropout{device,
                      hidden_shape,
                      dropout,
                      stream},
              dense_2{device,
                      hidden_shape,
                      input_shape,
                      optimizer_kw,
                      true,
                      stream} {}

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @note This function will also apply Dropout in training mode.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const cudaStream_t stream = core::default_stream) override {
            const auto &x1 = dense_1.forward(input, stream);
            const auto &x2 = relu.forward(x1, stream);
            const auto &x3 = dropout.forward(x2, stream);
            return dense_2.forward(x3, stream);
        }

        /**
         * @brief Compute the backward pass of the layer.
         *
         * @note This function will also update the parameters if the layer is
         * in training mode.
         *
         * @param error The error tensor from the next layer.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The gradient with respect to the input.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            const auto &x1 = dense_2.backward(error, stream);
            const auto &x2 = dropout.backward(x1, stream);
            const auto &x3 = relu.backward(x2, stream);
            return dense_1.backward(x3, stream);
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            return dense_2.get_output();
        }

        /**
         * @brief Get the gradient in respect to the input.
         */
        const core::Device_Tensor<data_type> &get_input_gradient()
            const override {
            return dense_1.get_input_gradient();
        }

        /**
         * @brief Get the first dense layer.
         */
        const Dense<data_type, optimizer_type> &get_dense_1() const {
            return dense_1;
        }

        /**
         * @brief Get the relu function.
         */
        const activations::ReLU<data_type> &get_relu() const { return relu; }

        /**
         * @brief Get the dropout layer.
         */
        const Dropout<data_type> &get_dropout() const { return dropout; }

        /**
         * @brief Get the second dense layer.
         */
        const Dense<data_type, optimizer_type> &get_dense_2() const {
            return dense_2;
        }

        /**
         * @brief Sets the layer into training mode.
         *
         * In training mode, dropout is applied in the forward pass, and in the
         * backward pass the trainable parameters will be updated.
         *
         * This layer is initially set to training mode.
         */
        void train() override {
            dense_1.train();
            dropout.train();
            dense_2.train();
        }

        /**
         * @brief Sets the layer into evaluation mode.
         *
         * In evaluation mode, the dropout behaves like the identity function
         * and the parameter update is omitted in the backward pass.
         *
         * This layer is initially set to training mode.
         */
        void eval() override {
            dense_1.eval();
            dropout.eval();
            dense_2.eval();
        }

        /**
         * @brief Returns the number of trainable parameters of the layer.
         */
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += dense_1.parameters();
            parameters += dense_2.parameters();
            return parameters;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += dense_1.mem_size();
            size += relu.mem_size();
            size += dropout.mem_size();
            size += dense_2.mem_size();
            return size;
        }

      private:
        /**
         * @brief Saves the parameters of the layer to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         */
        void _save(const std::string &path,
                   const std::string &filetype) override {
            dense_1.save(path + "_d1", filetype);
            dense_2.save(path + "_d2", filetype);
        }

        /**
         * @brief Loads the parameters of the layer from a file.
         *
         * @param path The base path to read the parameters from.
         * @param filetype The filetype of the saved files.
         * @param stream The CUDA stream used for copying from the host.
         */
        void _load(const std::string &path, const std::string &filetype,
                   const cudaStream_t &stream) override {
            dense_1.load(path + "_d1", filetype, stream);
            dense_2.load(path + "_d2", filetype, stream);
        }
    };

}  // namespace layers
