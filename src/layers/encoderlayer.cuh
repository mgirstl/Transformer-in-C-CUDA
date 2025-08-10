/**
 * @file encoderlayer.cuh
 * @brief Implements the EncoderLayer class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "../core.cuh"
#include "dropout.cuh"
#include "layer.cuh"
#include "layernorm.cuh"
#include "multiheadattention.cuh"
#include "positionwisefeedforward.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the EncoderLayer class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @class EncoderLayer
     * @brief The EncoderLayer layer.
     *
     * The EncoderLayer layer consists of a multi-head attention mechanism
     * followed by a position-wise feed-forward operation. This is a key
     * component of the Transformer architecture.
     *
     * For more information, see "Attention Is All You Need".
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
    class EncoderLayer final
        : public Layer<data_type, data_type, optimizer_type> {
      private:
        core::Device &device;
        MultiheadAttention<data_type, optimizer_type> mha;
        Dropout<data_type> mha_dropout;
        LayerNorm<data_type, optimizer_type> mha_norm;
        PositionwiseFeedForward<data_type, optimizer_type> pwff;
        Dropout<data_type> pwff_dropout;
        LayerNorm<data_type, optimizer_type> pwff_norm;
        core::Device_Tensor<data_type> added;  // multipurpose variable

      public:
        /**
         * @brief Construct a new EncoderLayer object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape (excluding the first axis) of the input/output
         * tensors, i.e., `[sequence_length, embedding_dim]`.
         * @param hidden_shape The shape (excluding the first axis) of the
         * features in the hidden dimension of the Positionwise Feed Forward
         * layer, i.e., `[sequence_length, hidden_dim]`.
         * @param optimizer_kw Will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param num_heads Number of parallel attention heads.
         * @param dropout The probability that an element in the dropout layer
         * is set to zero.
         * @param stream The CUDA stream used for launching the kernels.
         */
        EncoderLayer(
            core::Device &device, const std::vector<core::index_type> &shape,
            const std::vector<core::index_type> &hidden_shape,
            const std::unordered_map<std::string, data_type> &optimizer_kw = {},
            const core::index_type num_heads = 8, const data_type dropout = 0.1,
            const cudaStream_t stream = core::default_stream)
            : device{device},
              mha{device,    shape,   shape, optimizer_kw,
                  num_heads, dropout, stream},
              mha_dropout{device, shape, dropout, stream},
              mha_norm{device, shape, {shape.back()}, optimizer_kw, stream},
              pwff{device, shape, hidden_shape, optimizer_kw, dropout, stream},
              pwff_dropout{device, shape, dropout, stream},
              pwff_norm{device, shape, {shape.back()}, optimizer_kw, stream},
              added{core::NO_ALLOC, shape} {}

#pragma diag_suppress virtual_function_decl_hidden

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @note This function will also apply Dropout in training mode.
         *
         * @param input The input tensor. It has to have the shape
         * `[batchsize, sequence_length, embedding_dim]`.
         * @param mask The attention mask, i.e., the mask which specifies which
         * positions in the sequence get ignored in the attention calculation.
         * It has to have the shape `[batchsize, sequence_length]` or
         * `[batchsize, sequence_length, sequence_length]`.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const core::Device_Tensor<core::mask_type> &mask,
            const cudaStream_t stream = core::default_stream) {
            added.rebatchsize(input.batchsize(), stream);

            const auto &x1 = mha.forward(input, input, input, mask, stream);
            const auto &x2 = mha_dropout.forward(x1, stream);

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    input, x2, added);
            core::checkCuda(cudaPeekAtLastError());

            const auto &x3 = mha_norm.forward(added, stream);

            const auto &y1 = pwff.forward(x3, stream);
            const auto &y2 = pwff_dropout.forward(y1, stream);

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    x3, y2, added);
            core::checkCuda(cudaPeekAtLastError());

            return pwff_norm.forward(added, stream);
        }

#pragma diag_suppress virtual_function_decl_hidden

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
            added.rebatchsize(error.batchsize(), stream);

            const auto &x1 = pwff_norm.backward(error, stream);
            const auto &y1 = pwff_dropout.backward(x1, stream);
            const auto &y2 = pwff.backward(y1, stream);

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    x1, y2, added);
            core::checkCuda(cudaPeekAtLastError());

            const auto &z = mha_norm.backward(added, stream);
            const auto &w = mha_dropout.backward(z, stream);
            mha.backward(w, stream);

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    z, mha.get_query_gradient(), mha.get_key_gradient(),
                    mha.get_value_gradient(), added);
            core::checkCuda(cudaPeekAtLastError());

            return added;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            return pwff_norm.get_output();
        }

        /**
         * @brief Get the gradient in respect to the input.
         */
        const core::Device_Tensor<data_type> &get_input_gradient()
            const override {
            return added;
        }

        /**
         * @brief Get the MultiheadAttention layer.
         */
        const MultiheadAttention<data_type, optimizer_type> &get_mha() const {
            return mha;
        }

        /**
         * @brief Get the Dropout layer which is after the MultiheadAttention
         * layer.
         */
        const Dropout<data_type> &get_mha_dropout() const {
            return mha_dropout;
        }

        /**
         * @brief Get the LayerNorm layer which is after the MultiheadAttention
         * layer.
         */
        const LayerNorm<data_type, optimizer_type> &get_mha_norm() const {
            return mha_norm;
        }

        /**
         * @brief Get the PositionwiseFeedForward layer.
         */
        const PositionwiseFeedForward<data_type, optimizer_type> &get_pwff()
            const {
            return pwff;
        }

        /**
         * @brief Get the Dropout layer which is after the
         * PositionwiseFeedForward layer.
         */
        const Dropout<data_type> &get_pwff_dropout() const {
            return pwff_dropout;
        }

        /**
         * @brief Get the LayerNorm layer which is after the
         * PositionwiseFeedForward layer.
         */
        const LayerNorm<data_type, optimizer_type> &get_pwff_norm() const {
            return pwff_norm;
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
            mha.train();
            mha_dropout.train();
            mha_norm.train();
            pwff.train();
            pwff_dropout.train();
            pwff_norm.train();
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
            mha.eval();
            mha_dropout.eval();
            mha_norm.eval();
            pwff.eval();
            pwff_dropout.eval();
            pwff_norm.eval();
        }

        /**
         * @brief Returns the number of trainable parameters of the layer.
         */
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += mha.parameters();
            parameters += mha_norm.parameters();
            parameters += pwff.parameters();
            parameters += pwff_norm.parameters();
            return parameters;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += mha.mem_size();
            size += mha_dropout.mem_size();
            size += mha_norm.mem_size();
            size += pwff.mem_size();
            size += pwff_dropout.mem_size();
            size += pwff_norm.mem_size();
            size += added.mem_size();
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
            mha.save(path + "_mha", filetype);
            mha_norm.save(path + "_mha", filetype);
            pwff.save(path + "_pwff", filetype);
            pwff_norm.save(path + "_pwff", filetype);
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
            mha.load(path + "_mha", filetype, stream);
            mha_norm.load(path + "_mha", filetype, stream);
            pwff.load(path + "_pwff", filetype, stream);
            pwff_norm.load(path + "_pwff", filetype, stream);
        }
    };

}  // namespace layers
