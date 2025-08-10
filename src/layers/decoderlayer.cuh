/**
 * @file decoderlayer.cuh
 * @brief Implements the DecoderLayer class.
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
 * This specific file defines the DecoderLayer class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @class DecoderLayer
     * @brief The DecoderLayer layer.
     *
     * The DecoderLayer layer consists of a multi-head attention mechanism over
     * the target, a multi-head attention mechanism over the encoder's output,
     * and a position-wise feed-forward network. This is a key component of the
     * Transformer architecture.
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
    class DecoderLayer final
        : public Layer<data_type, data_type, optimizer_type> {
      private:
        core::Device &device;
        MultiheadAttention<data_type, optimizer_type> mha_1;
        Dropout<data_type> mha_1_dropout;
        LayerNorm<data_type, optimizer_type> mha_1_norm;
        MultiheadAttention<data_type, optimizer_type> mha_2;
        Dropout<data_type> mha_2_dropout;
        LayerNorm<data_type, optimizer_type> mha_2_norm;
        PositionwiseFeedForward<data_type, optimizer_type> pwff;
        Dropout<data_type> pwff_dropout;
        LayerNorm<data_type, optimizer_type> pwff_norm;
        core::Device_Tensor<data_type> added;  // multipurpose variable
        core::Device_Tensor<data_type> source_gradient;
        core::Device_Tensor<data_type> target_gradient;
        core::Device_Tensor<data_type> null_tensor;

      public:
        /**
         * @brief Construct a new DecoderLayer object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param target_shape The shape (excluding the first axis) of the main
         * input and output, i.e., `[target_sequence_length, embedding_dim]`.
         * @param source_shape The shape (excluding the first axis) of the
         * second input, i.e., `[source_sequence_length, embedding_dim]`.
         * @param hidden_shape The shape (excluding the first axis) of the
         * features in the hidden dimension of the Positionwise Feed Forward
         * layer, i.e., `[target_sequence_length, hidden_dim]`.
         * @param optimizer_kw Will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param num_heads Number of parallel attention heads.
         * @param dropout The probability that an element in the dropout layer
         * is set to zero.
         * @param stream The CUDA stream used for launching the kernels.
         */
        DecoderLayer(
            core::Device &device,
            const std::vector<core::index_type> &target_shape,
            const std::vector<core::index_type> &source_shape,
            const std::vector<core::index_type> &hidden_shape,
            const std::unordered_map<std::string, data_type> &optimizer_kw = {},
            const core::index_type num_heads = 8, const data_type dropout = 0.1,
            const cudaStream_t stream = core::default_stream)
            : device{device},
              mha_1{device,
                    target_shape,
                    target_shape,
                    optimizer_kw,
                    num_heads,
                    dropout,
                    stream},
              mha_1_dropout{device, target_shape, dropout, stream},
              mha_1_norm{device,
                         target_shape,
                         {target_shape.back()},
                         optimizer_kw,
                         stream},
              mha_2{device,
                    target_shape,
                    source_shape,
                    optimizer_kw,
                    num_heads,
                    dropout,
                    stream},
              mha_2_dropout{device, target_shape, dropout, stream},
              mha_2_norm{device,
                         target_shape,
                         {target_shape.back()},
                         optimizer_kw,
                         stream},
              pwff{device,
                   target_shape,
                   hidden_shape,
                   optimizer_kw,
                   dropout,
                   stream},
              pwff_dropout{device, target_shape, dropout, stream},
              pwff_norm{device,
                        target_shape,
                        {target_shape.back()},
                        optimizer_kw,
                        stream},
              added{core::NO_ALLOC, target_shape},
              source_gradient{core::NO_ALLOC, source_shape},
              target_gradient{core::NO_ALLOC, target_shape} {}

#pragma diag_suppress virtual_function_decl_hidden

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @note This function will also apply Dropout in training mode.
         *
         * @param target The main input tensor. It has to have the shape
         * `[batchsize, target_sequence_length, embedding_dim]`.
         * @param target_mask The target attention mask, i.e., the mask which
         * specifies which positions in the sequence get ignored in the
         * attention calculation. It has to have the shape
         * `[batchsize, target_sequence_length, target_sequence_length]` or
         * `[batchsize, target_sequence_length]`.
         * @param source The output of the Encoder layer. It has to have the
         * shape `[batchsize, source_sequence_length, embedding_dim]`.
         * @param source_mask The source attention mask, i.e., the mask which
         * specifies which positions in the sequence get ignored in the
         * attention calculation. It has to have the shape
         * `[batchsize, source_sequence_length, source_sequence_length]` or
         * `[batchsize, source_sequence_length]`.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &target,
            const core::Device_Tensor<core::mask_type> &target_mask,
            const core::Device_Tensor<data_type> &source,
            const core::Device_Tensor<core::mask_type> &source_mask,
            const cudaStream_t stream = core::default_stream) {
            added.rebatchsize(target.batchsize(), stream);

            const auto &x1 =
                mha_1.forward(target, target, target, target_mask, stream);
            const auto &x2 = mha_1_dropout.forward(x1, stream);

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    target, x2, added);
            core::checkCuda(cudaPeekAtLastError());

            const auto &x3 = mha_1_norm.forward(added, stream);

            const auto &y1 =
                mha_2.forward(x3, source, source, source_mask, stream);
            const auto &y2 = mha_2_dropout.forward(y1, stream);

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    x3, y2, added);
            core::checkCuda(cudaPeekAtLastError());

            const auto &z = mha_2_norm.forward(added, stream);

            const auto &w1 = pwff.forward(z, stream);
            const auto &w2 = pwff_dropout.forward(w1, stream);

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    z, w2, added);
            core::checkCuda(cudaPeekAtLastError());

            return pwff_norm.forward(added, stream);
        }

#pragma diag_default virtual_function_decl_hidden

        /**
         * @brief Compute the backward pass of the layer.
         *
         * @note This function will also update the parameters if the layer is
         * in training mode.
         *
         * @param error The error tensor from the next layer.
         * @param stream The CUDA stream used for launching the kernels.
         * @return An empty tensor.
         *
         * @attention This function returns an empty tensor instead of the
         * `input_gradient`. The reason for this is that the forward pass has
         * multiple input tensors.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            // preparation
            added.rebatchsize(error.batchsize(), stream);
            target_gradient.rebatchsize(error.batchsize(), stream);
            source_gradient.rebatchsize(error.batchsize(), stream);

            // calculate the gradient in respect to the input of the second
            // MultiheadAttention layer.
            const auto &x1 = pwff_norm.backward(error, stream);
            const auto &y1 = pwff_dropout.backward(x1, stream);
            const auto &y2 = pwff.backward(y1, stream);

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    x1, y2, added);
            core::checkCuda(cudaPeekAtLastError());

            const auto &x2 = mha_2_norm.backward(added, stream);

            const auto &z1 = mha_2_dropout.backward(x2, stream);
            mha_2.backward(z1, stream);

            // calculate the gradient in respect to the source
            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    mha_2.get_key_gradient(), mha_2.get_value_gradient(),
                    source_gradient);
            core::checkCuda(cudaPeekAtLastError());

            // calculate the gradient in respect to the target
            const auto &z2 = mha_2.get_query_gradient();

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    x2, z2, added);
            core::checkCuda(cudaPeekAtLastError());

            const auto &x3 = mha_1_norm.backward(added, stream);

            const auto &z3 = mha_1_dropout.backward(x3, stream);
            mha_1.backward(z3, stream);

            core::add<data_type>
                <<<device.blocks(added.size()), device.threads(), 0, stream>>>(
                    x3, mha_1.get_query_gradient(), mha_1.get_key_gradient(),
                    mha_1.get_value_gradient(), target_gradient);
            core::checkCuda(cudaPeekAtLastError());

            return null_tensor;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            return pwff_norm.get_output();
        }

        /**
         * @brief Get the in the forward pass calculated attention.
         */
        const core::Device_Tensor<data_type> &get_attention() const {
            return mha_2.get_attention();
        }

        /**
         * @brief Get the gradient in respect to the target.
         */
        const core::Device_Tensor<data_type> &get_target_gradient() const {
            return target_gradient;
        }

        /**
         * @brief Get the gradient in respect to the source.
         */
        const core::Device_Tensor<data_type> &get_source_gradient() const {
            return source_gradient;
        }

        /**
         * @brief Get the first MultiheadAttention layer.
         */
        const MultiheadAttention<data_type, optimizer_type> &get_mha_1() const {
            return mha_1;
        }

        /**
         * @brief Get the Dropout layer which is after the first
         * MultiheadAttention layer.
         */
        const Dropout<data_type> &get_mha_1_dropout() const {
            return mha_1_dropout;
        }

        /**
         * @brief Get the LayerNorm layer which is after the first
         * MultiheadAttention layer.
         */
        const LayerNorm<data_type, optimizer_type> &get_mha_1_norm() const {
            return mha_1_norm;
        }

        /**
         * @brief Get the second MultiheadAttention layer.
         */
        const MultiheadAttention<data_type, optimizer_type> &get_mha_2() const {
            return mha_2;
        }

        /**
         * @brief Get the Dropout layer which is after the second
         * MultiheadAttention layer.
         */
        const Dropout<data_type> &get_mha_2_dropout() const {
            return mha_2_dropout;
        }

        /**
         * @brief Get the LayerNorm layer which is after the second
         * MultiheadAttention layer.
         */
        const LayerNorm<data_type, optimizer_type> &get_mha_2_norm() const {
            return mha_2_norm;
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
            mha_1.train();
            mha_1_dropout.train();
            mha_1_norm.train();
            mha_2.train();
            mha_2_dropout.train();
            mha_2_norm.train();
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
            mha_1.eval();
            mha_1_dropout.eval();
            mha_1_norm.eval();
            mha_2.eval();
            mha_2_dropout.eval();
            mha_2_norm.eval();
            pwff.eval();
            pwff_dropout.eval();
            pwff_norm.eval();
        }

        /**
         * @brief Returns the number of trainable parameters of the layer.
         */
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += mha_1.parameters();
            parameters += mha_1_norm.parameters();
            parameters += mha_2.parameters();
            parameters += mha_2_norm.parameters();
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
            size += mha_1.mem_size();
            size += mha_1_dropout.mem_size();
            size += mha_1_norm.mem_size();
            size += mha_2.mem_size();
            size += mha_2_dropout.mem_size();
            size += mha_2_norm.mem_size();
            size += pwff.mem_size();
            size += pwff_dropout.mem_size();
            size += pwff_norm.mem_size();
            size += added.mem_size();
            size += target_gradient.mem_size();
            size += source_gradient.mem_size();
            size += null_tensor.mem_size(true);
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
            mha_1.save(path + "_mha1", filetype);
            mha_1_norm.save(path + "_mha1", filetype);
            mha_2.save(path + "_mha2", filetype);
            mha_2_norm.save(path + "_mha2", filetype);
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
            mha_1.load(path + "_mha1", filetype, stream);
            mha_1_norm.load(path + "_mha1", filetype, stream);
            mha_2.load(path + "_mha2", filetype, stream);
            mha_2_norm.load(path + "_mha2", filetype, stream);
            pwff.load(path + "_pwff", filetype, stream);
            pwff_norm.load(path + "_pwff", filetype, stream);
        }
    };

}  // namespace layers
